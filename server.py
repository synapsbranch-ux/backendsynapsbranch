from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import asyncio
import re
import json
import hashlib
import secrets
from passlib.context import CryptContext
from jose import jwt, JWTError
import httpx

# Import settings (Pydantic Settings handles .env loading automatically)
from settings import settings

# Import our LLM service
from llm_service import get_llm_service, AVAILABLE_MODELS

# MongoDB connection (uses validated settings)
client = AsyncIOMotorClient(settings.MONGO_URL)
db = client[settings.DB_NAME]

# OAuth state storage (in production, use Redis)
oauth_states = {}

app = FastAPI(title="SynapsBranch API", version="2.0.0")
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============= MODELS =============

class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    color: Optional[str] = "#6366f1"
    icon: Optional[str] = "folder"
    system_prompt: Optional[str] = "You are a helpful AI assistant."

class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
    system_prompt: Optional[str] = None

class Workspace(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    description: str
    color: str
    icon: str
    system_prompt: str
    created_at: str
    updated_at: str

class ConversationCreate(BaseModel):
    workspace_id: Optional[str] = None
    title: Optional[str] = "New Chat"

class Conversation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    workspace_id: Optional[str]
    title: str
    created_at: str
    updated_at: str

class MessageCreate(BaseModel):
    conversation_id: str
    parent_id: Optional[str] = None
    content: str
    role: str = "user"
    branch_name: str = "main"
    model_used: Optional[str] = "gpt-4o-mini"

class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    conversation_id: str
    parent_id: Optional[str]
    content: str
    role: str
    branch_name: str
    model_used: Optional[str]
    created_at: str

class ForkRequest(BaseModel):
    new_branch_name: str
    new_content: str

class ChatRequest(BaseModel):
    conversation_id: str
    parent_id: Optional[str] = None
    content: str
    branch_name: str = "main"
    model: str = "gpt-4o-mini"
    system_prompt: Optional[str] = None
    canvas_context: Optional[Dict[str, Any]] = None  # {content, language, isOpen}

class BranchInfo(BaseModel):
    name: str
    message_count: int
    head_message_id: Optional[str]

class FileDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    workspace_id: str
    filename: str
    file_type: str
    content: str
    created_at: str

class Artifact(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    conversation_id: Optional[str]
    message_id: Optional[str]
    type: str  # code, mermaid, svg, csv, json
    content: str
    created_at: str

class CanvasCreate(BaseModel):
    content: str
    language: str = "html"
    workspace_id: Optional[str] = None
    conversation_id: Optional[str] = None

class CanvasVersionCreate(BaseModel):
    content: str
    language: Optional[str] = None

class CanvasDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    content: str
    language: str
    workspace_id: Optional[str]
    conversation_id: Optional[str]
    versions: List[Dict[str, Any]]
    created_at: str
    updated_at: str

# ============= WORKSPACE ENDPOINTS =============

@api_router.post("/workspaces", response_model=Workspace, status_code=201)
async def create_workspace(input: WorkspaceCreate):
    now = datetime.now(timezone.utc).isoformat()
    workspace = {
        "id": str(uuid.uuid4()),
        "name": input.name,
        "description": input.description or "",
        "color": input.color or "#6366f1",
        "icon": input.icon or "folder",
        "system_prompt": input.system_prompt or "You are a helpful AI assistant.",
        "created_at": now,
        "updated_at": now
    }
    await db.workspaces.insert_one(workspace)
    return Workspace(**workspace)

@api_router.get("/workspaces", response_model=List[Workspace])
async def get_workspaces():
    workspaces = await db.workspaces.find({}, {"_id": 0}).to_list(1000)
    return [Workspace(**w) for w in workspaces]

@api_router.get("/workspaces/{workspace_id}", response_model=Workspace)
async def get_workspace(workspace_id: str):
    workspace = await db.workspaces.find_one({"id": workspace_id}, {"_id": 0})
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return Workspace(**workspace)

@api_router.put("/workspaces/{workspace_id}", response_model=Workspace)
async def update_workspace(workspace_id: str, input: WorkspaceUpdate):
    updates = {k: v for k, v in input.model_dump().items() if v is not None}
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    await db.workspaces.update_one({"id": workspace_id}, {"$set": updates})
    workspace = await db.workspaces.find_one({"id": workspace_id}, {"_id": 0})
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return Workspace(**workspace)

@api_router.delete("/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    result = await db.workspaces.delete_one({"id": workspace_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Workspace not found")
    # Delete associated conversations and messages
    convs = await db.conversations.find({"workspace_id": workspace_id}).to_list(1000)
    conv_ids = [c["id"] for c in convs]
    await db.messages.delete_many({"conversation_id": {"$in": conv_ids}})
    await db.conversations.delete_many({"workspace_id": workspace_id})
    await db.files.delete_many({"workspace_id": workspace_id})
    return {"message": "Workspace deleted"}

# ============= AUTH MODELS =============

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    email: EmailStr
    name: Optional[str]
    avatar: Optional[str] = None
    provider: str = "email"
    access_granted: bool = False
    created_at: str

class Token(BaseModel):
    access_token: str
    token_type: str

class InviteCodeValidate(BaseModel):
    code: str

class InviteCode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    code: str
    is_used: bool = False
    used_by: Optional[str] = None
    created_at: str

# ============= AUTH UTILS =============

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    
    user = await db.users.find_one({"email": email}, {"_id": 0})
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user)

# ============= AUTH ENDPOINTS =============

@api_router.post("/auth/register", response_model=User, status_code=201)
async def register(user_input: UserCreate):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_input.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    now = datetime.now(timezone.utc).isoformat()
    hashed_password = get_password_hash(user_input.password)
    
    user_doc = {
        "id": str(uuid.uuid4()),
        "email": user_input.email,
        "name": user_input.name or user_input.email.split('@')[0],
        "hashed_password": hashed_password,
        "provider": "email",
        "provider_id": None,
        "avatar": None,
        "access_granted": False,  # New users need invite code
        "created_at": now,
        "updated_at": now
    }
    
    await db.users.insert_one(user_doc)
    return User(**user_doc)

@api_router.post("/auth/login", response_model=Token)
async def login(user_input: UserLogin):
    user = await db.users.find_one({"email": user_input.email})
    if not user or not user.get("hashed_password") or not verify_password(user_input.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@api_router.get("/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# ============= OAUTH ENDPOINTS =============

@api_router.get("/auth/google")
async def google_auth():
    """Initiate Google OAuth flow"""
    if not settings.google_oauth_enabled:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"provider": "google", "created_at": datetime.now(timezone.utc)}
    
    redirect_uri = f"{settings.FRONTEND_URL}/auth/callback"
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={settings.GOOGLE_CLIENT_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope=openid%20email%20profile&"
        f"state={state}"
    )
    return {"auth_url": google_auth_url}

@api_router.get("/auth/google/callback")
async def google_callback(code: str, state: str):
    """Handle Google OAuth callback"""
    if state not in oauth_states or oauth_states[state]["provider"] != "google":
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    
    del oauth_states[state]
    
    # Exchange code for tokens
    redirect_uri = f"{settings.FRONTEND_URL}/auth/callback"
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code"
            }
        )
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code for token")
        
        tokens = token_response.json()
        
        # Get user info
        user_response = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info")
        
        google_user = user_response.json()
    
    # Find or create user
    email = google_user["email"]
    existing_user = await db.users.find_one({"email": email})
    
    now = datetime.now(timezone.utc).isoformat()
    
    if existing_user:
        # Update existing user with Google info
        await db.users.update_one(
            {"email": email},
            {"$set": {
                "provider": "google",
                "provider_id": google_user["id"],
                "avatar": google_user.get("picture"),
                "name": google_user.get("name", existing_user.get("name")),
                "updated_at": now
            }}
        )
        user = await db.users.find_one({"email": email}, {"_id": 0})
    else:
        # Create new user
        user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "name": google_user.get("name", email.split('@')[0]),
            "hashed_password": None,
            "provider": "google",
            "provider_id": google_user["id"],
            "avatar": google_user.get("picture"),
            "access_granted": False,  # New users need invite code
            "created_at": now,
            "updated_at": now
        }
        await db.users.insert_one(user)
    
    # Generate JWT
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer", "user": User(**user)}

@api_router.get("/auth/github")
async def github_auth():
    """Initiate GitHub OAuth flow"""
    if not settings.github_oauth_enabled:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured")
    
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"provider": "github", "created_at": datetime.now(timezone.utc)}
    
    redirect_uri = f"{settings.FRONTEND_URL}/auth/callback"
    github_auth_url = (
        f"https://github.com/login/oauth/authorize?"
        f"client_id={settings.GITHUB_CLIENT_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"scope=user:email&"
        f"state={state}"
    )
    return {"auth_url": github_auth_url}

@api_router.get("/auth/github/callback")
async def github_callback(code: str, state: str):
    """Handle GitHub OAuth callback"""
    if state not in oauth_states or oauth_states[state]["provider"] != "github":
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    
    del oauth_states[state]
    
    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "code": code,
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
            },
            headers={"Accept": "application/json"}
        )
        
        if token_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to exchange code for token")
        
        tokens = token_response.json()
        access_token = tokens.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token received")
        
        # Get user info
        user_response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        
        if user_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info")
        
        github_user = user_response.json()
        
        # Get email (may need separate call if private)
        email = github_user.get("email")
        if not email:
            emails_response = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if emails_response.status_code == 200:
                emails = emails_response.json()
                primary_email = next((e for e in emails if e.get("primary")), None)
                email = primary_email["email"] if primary_email else emails[0]["email"]
    
    if not email:
        raise HTTPException(status_code=400, detail="Could not get email from GitHub")
    
    # Find or create user
    existing_user = await db.users.find_one({"email": email})
    
    now = datetime.now(timezone.utc).isoformat()
    
    if existing_user:
        # Update existing user with GitHub info
        await db.users.update_one(
            {"email": email},
            {"$set": {
                "provider": "github",
                "provider_id": str(github_user["id"]),
                "avatar": github_user.get("avatar_url"),
                "name": github_user.get("name") or github_user.get("login") or existing_user.get("name"),
                "updated_at": now
            }}
        )
        user = await db.users.find_one({"email": email}, {"_id": 0})
    else:
        # Create new user
        user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "name": github_user.get("name") or github_user.get("login") or email.split('@')[0],
            "hashed_password": None,
            "provider": "github",
            "provider_id": str(github_user["id"]),
            "avatar": github_user.get("avatar_url"),
            "access_granted": False,  # New users need invite code
            "created_at": now,
            "updated_at": now
        }
        await db.users.insert_one(user)
    
    # Generate JWT
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    jwt_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    
    return {"access_token": jwt_token, "token_type": "bearer", "user": User(**user)}

# ============= INVITE CODE ENDPOINTS =============

@api_router.post("/auth/validate-invite")
async def validate_invite_code(input: InviteCodeValidate, current_user: User = Depends(get_current_user)):
    """Validate and redeem an invite code"""
    # Check if user already has access
    user = await db.users.find_one({"email": current_user.email}, {"_id": 0})
    if user and user.get("access_granted"):
        return {"success": True, "message": "Access already granted"}
    
    # Find the invite code
    invite_code = await db.invite_codes.find_one({"code": input.code.upper()}, {"_id": 0})
    
    if not invite_code:
        raise HTTPException(status_code=400, detail="Invalid invite code")
    
    if invite_code.get("is_used"):
        raise HTTPException(status_code=400, detail="This invite code has already been used")
    
    # Mark code as used and grant access
    now = datetime.now(timezone.utc).isoformat()
    
    await db.invite_codes.update_one(
        {"code": input.code.upper()},
        {"$set": {"is_used": True, "used_by": current_user.id, "used_at": now}}
    )
    
    await db.users.update_one(
        {"email": current_user.email},
        {"$set": {"access_granted": True, "updated_at": now}}
    )
    
    return {"success": True, "message": "Access granted successfully"}

# NOTE: Invite code generation has been moved to a secure CLI script
# Run: python generate_invite_codes.py --count 5

# ============= CONVERSATION ENDPOINTS =============

@api_router.post("/conversations", response_model=Conversation, status_code=201)
async def create_conversation(input: ConversationCreate):
    now = datetime.now(timezone.utc).isoformat()
    conversation = {
        "id": str(uuid.uuid4()),
        "workspace_id": input.workspace_id,
        "title": input.title or "New Chat",
        "created_at": now,
        "updated_at": now
    }
    await db.conversations.insert_one(conversation)
    return Conversation(**conversation)

@api_router.get("/conversations", response_model=List[Conversation])
async def get_conversations(workspace_id: Optional[str] = None, standalone: Optional[bool] = None):
    query = {}
    if workspace_id:
        query["workspace_id"] = workspace_id
    elif standalone:
        # Only return conversations without a workspace (standalone chats)
        query["workspace_id"] = None
    conversations = await db.conversations.find(query, {"_id": 0}).sort("updated_at", -1).to_list(1000)
    return [Conversation(**c) for c in conversations]

@api_router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    conversation = await db.conversations.find_one({"id": conversation_id}, {"_id": 0})
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Conversation(**conversation)

@api_router.put("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, title: str):
    now = datetime.now(timezone.utc).isoformat()
    await db.conversations.update_one({"id": conversation_id}, {"$set": {"title": title, "updated_at": now}})
    return {"message": "Updated"}

@api_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    result = await db.conversations.delete_one({"id": conversation_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.messages.delete_many({"conversation_id": conversation_id})
    return {"message": "Conversation deleted"}

# ============= MESSAGE ENDPOINTS =============

@api_router.post("/messages", response_model=Message, status_code=201)
async def create_message(input: MessageCreate):
    now = datetime.now(timezone.utc).isoformat()
    message = {
        "id": str(uuid.uuid4()),
        "conversation_id": input.conversation_id,
        "parent_id": input.parent_id,
        "content": input.content,
        "role": input.role,
        "branch_name": input.branch_name,
        "model_used": input.model_used,
        "created_at": now
    }
    await db.messages.insert_one(message)
    await db.conversations.update_one({"id": input.conversation_id}, {"$set": {"updated_at": now}})
    return Message(**message)

@api_router.get("/messages", response_model=List[Message])
async def get_messages(conversation_id: str, branch_name: Optional[str] = None):
    query = {"conversation_id": conversation_id}
    if branch_name:
        query["branch_name"] = branch_name
    messages = await db.messages.find(query, {"_id": 0}).sort("created_at", 1).to_list(10000)
    return [Message(**m) for m in messages]

@api_router.get("/messages/{message_id}", response_model=Message)
async def get_message(message_id: str):
    message = await db.messages.find_one({"id": message_id}, {"_id": 0})
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return Message(**message)

@api_router.delete("/messages/{message_id}")
async def delete_message(message_id: str):
    result = await db.messages.delete_one({"id": message_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"message": "Deleted"}

@api_router.get("/messages/{message_id}/path", response_model=List[Message])
async def get_message_path(message_id: str):
    """Get all messages from root to this message (for context)"""
    path = []
    current_id = message_id
    while current_id:
        msg = await db.messages.find_one({"id": current_id}, {"_id": 0})
        if not msg:
            break
        path.append(Message(**msg))
        current_id = msg.get("parent_id")
    path.reverse()
    return path

@api_router.post("/messages/{message_id}/fork", response_model=Message, status_code=201)
async def fork_message(message_id: str, input: ForkRequest):
    """Fork from a message to create a new branch - inherits full conversation history"""
    parent_msg = await db.messages.find_one({"id": message_id}, {"_id": 0})
    if not parent_msg:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Get the full path from root to this message
    path = []
    current_id = message_id
    while current_id:
        msg = await db.messages.find_one({"id": current_id}, {"_id": 0})
        if not msg:
            break
        path.append(msg)
        current_id = msg.get("parent_id")
    path.reverse()
    
    # Copy all messages in path to the new branch (except the last one, which is the fork point)
    now = datetime.now(timezone.utc).isoformat()
    prev_id = None
    for i, msg in enumerate(path):
        new_msg = {
            "id": str(uuid.uuid4()),
            "conversation_id": parent_msg["conversation_id"],
            "parent_id": prev_id,
            "content": msg["content"],
            "role": msg["role"],
            "branch_name": input.new_branch_name,
            "model_used": msg.get("model_used"),
            "created_at": now
        }
        await db.messages.insert_one(new_msg)
        prev_id = new_msg["id"]
    
    # Now add the new user message
    new_message = {
        "id": str(uuid.uuid4()),
        "conversation_id": parent_msg["conversation_id"],
        "parent_id": prev_id,
        "content": input.new_content,
        "role": "user",
        "branch_name": input.new_branch_name,
        "model_used": None,
        "created_at": now
    }
    await db.messages.insert_one(new_message)
    await db.conversations.update_one({"id": parent_msg["conversation_id"]}, {"$set": {"updated_at": now}})
    return Message(**new_message)

# ============= BRANCH ENDPOINTS =============

@api_router.get("/conversations/{conversation_id}/branches", response_model=List[BranchInfo])
async def get_branches(conversation_id: str):
    """Get all branches for a conversation"""
    pipeline = [
        {"$match": {"conversation_id": conversation_id}},
        {"$group": {
            "_id": "$branch_name",
            "count": {"$sum": 1},
            "last_message": {"$last": "$id"}
        }}
    ]
    results = await db.messages.aggregate(pipeline).to_list(100)
    branches = []
    for r in results:
        branches.append(BranchInfo(
            name=r["_id"],
            message_count=r["count"],
            head_message_id=r["last_message"]
        ))
    return branches

# ============= CHAT ENDPOINT (AI) =============

def extract_artifacts(content: str, conversation_id: str, message_id: str) -> List[dict]:
    """Extract code blocks, mermaid diagrams, etc. from AI response"""
    artifacts = []
    now = datetime.now(timezone.utc).isoformat()
    
    # Extract code blocks
    code_pattern = r'```(\w+)?\n([\s\S]*?)```'
    matches = re.findall(code_pattern, content)
    for lang, code in matches:
        artifact_type = 'mermaid' if lang and lang.lower() == 'mermaid' else 'code'
        artifacts.append({
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "message_id": message_id,
            "type": artifact_type,
            "content": code.strip(),
            "language": lang or "text",
            "created_at": now
        })
    
    return artifacts

@api_router.post("/chat")
async def chat(request: ChatRequest):
    """Send a message and get AI response"""
    try:
        # Save user message
        now = datetime.now(timezone.utc).isoformat()
        user_message = {
            "id": str(uuid.uuid4()),
            "conversation_id": request.conversation_id,
            "parent_id": request.parent_id,
            "content": request.content,
            "role": "user",
            "branch_name": request.branch_name,
            "model_used": None,
            "created_at": now
        }
        await db.messages.insert_one(user_message)
        
        # Get context path
        context_messages = []
        if request.parent_id:
            path = []
            current_id = request.parent_id
            while current_id:
                msg = await db.messages.find_one({"id": current_id}, {"_id": 0})
                if not msg:
                    break
                path.append(msg)
                current_id = msg.get("parent_id")
            path.reverse()
            for p in path:
                context_messages.append({"role": p["role"], "content": p["content"]})
        
        # Add current message
        context_messages.append({"role": "user", "content": request.content})
        
        # Get system prompt
        system_prompt = request.system_prompt or "You are a helpful AI assistant. Format your responses using Markdown when appropriate. When writing code, always use code blocks with language specification."
        
        # Check if conversation belongs to a workspace
        conv = await db.conversations.find_one({"id": request.conversation_id}, {"_id": 0})
        if conv and conv.get("workspace_id"):
            workspace = await db.workspaces.find_one({"id": conv["workspace_id"]}, {"_id": 0})
            if workspace and workspace.get("system_prompt"):
                system_prompt = workspace["system_prompt"]
            
            # Add file context (RAG)
            files = await db.files.find({"workspace_id": conv["workspace_id"]}, {"_id": 0}).to_list(10)
            if files:
                file_context = "\n\nContext from uploaded files:\n"
                for f in files:
                    file_context += f"\n--- {f['filename']} ---\n{f['content'][:2000]}\n"
                system_prompt += file_context
        
        # Call LLM using new service
        llm_service = get_llm_service()
        response = await llm_service.chat(
            messages=context_messages,
            system_prompt=system_prompt,
            model=request.model
        )
        
        # Save AI response
        ai_message_id = str(uuid.uuid4())
        ai_message = {
            "id": ai_message_id,
            "conversation_id": request.conversation_id,
            "parent_id": user_message["id"],
            "content": response,
            "role": "assistant",
            "branch_name": request.branch_name,
            "model_used": request.model,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.messages.insert_one(ai_message)
        
        # Extract and save artifacts
        artifacts = extract_artifacts(response, request.conversation_id, ai_message_id)
        if artifacts:
            await db.artifacts.insert_many(artifacts)
        
        # Update conversation
        await db.conversations.update_one(
            {"id": request.conversation_id},
            {"$set": {"updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        
        # Auto-title if first message
        msg_count = await db.messages.count_documents({"conversation_id": request.conversation_id})
        if msg_count <= 2:
            title = request.content[:50] + ("..." if len(request.content) > 50 else "")
            await db.conversations.update_one(
                {"id": request.conversation_id},
                {"$set": {"title": title}}
            )
        
        return {
            "user_message": Message(**user_message),
            "ai_message": Message(**ai_message)
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream AI response using SSE"""
    try:
        # Save user message
        now = datetime.now(timezone.utc).isoformat()
        user_message_id = str(uuid.uuid4())
        user_message = {
            "id": user_message_id,
            "conversation_id": request.conversation_id,
            "parent_id": request.parent_id,
            "content": request.content,
            "role": "user",
            "branch_name": request.branch_name,
            "model_used": None,
            "created_at": now
        }
        await db.messages.insert_one(user_message)

        # Get context path
        context_messages = []
        if request.parent_id:
            path = []
            current_id = request.parent_id
            while current_id:
                msg = await db.messages.find_one({"id": current_id}, {"_id": 0})
                if not msg:
                    break
                path.append(msg)
                current_id = msg.get("parent_id")
            path.reverse()
            for p in path:
                context_messages.append({"role": p["role"], "content": p["content"]})
        
        # Add current message
        context_messages.append({"role": "user", "content": request.content})

        # Get system prompt
        system_prompt = request.system_prompt or "You are a helpful AI assistant. Format your responses using Markdown when appropriate. When writing code, always use code blocks with language specification."
        
        # Check if conversation belongs to a workspace
        conv = await db.conversations.find_one({"id": request.conversation_id}, {"_id": 0})
        if conv and conv.get("workspace_id"):
            workspace = await db.workspaces.find_one({"id": conv["workspace_id"]}, {"_id": 0})
            if workspace and workspace.get("system_prompt"):
                system_prompt = workspace["system_prompt"]
            
            # Add file context (RAG)
            files = await db.files.find({"workspace_id": conv["workspace_id"]}, {"_id": 0}).to_list(10)
            if files:
                file_context = "\n\nContext from uploaded files:\n"
                for f in files:
                    file_context += f"\n--- {f['filename']} ---\n{f['content'][:2000]}\n"
                system_prompt += file_context

        # Add Canvas Tool capability if canvas is open
        if request.canvas_context and request.canvas_context.get("isOpen"):
            canvas_instructions = """

=== CANVAS TOOL ===
The user has a Canvas (code editor) open. You can write code DIRECTLY to the canvas.

To write code to the canvas, use this exact XML format:
<canvas lang="LANGUAGE">
YOUR CODE HERE
</canvas>

Supported languages: html, css, javascript, typescript, python, jsx, tsx, json, markdown, mermaid

RULES:
1. Keep explanations OUTSIDE the canvas tags - they appear in chat
2. Put ALL code INSIDE the canvas tags - it streams to the editor
3. When modifying existing code, output the COMPLETE updated file
4. Use the appropriate language tag for syntax highlighting

Example response:
"Here's a React component for you:
<canvas lang="jsx">
import React from 'react';

export default function Button({ label }) {
  return <button className="btn">{label}</button>;
}
</canvas>
This component accepts a label prop and renders a styled button."
"""
            if request.canvas_context.get("content"):
                canvas_instructions += f"""

=== CURRENT CANVAS CONTENT ({request.canvas_context.get('language', 'html').upper()}) ===
```
{request.canvas_context.get('content', '')[:5000]}
```
When the user asks to modify or update, use this as the base and output the complete modified version.
"""
            system_prompt += canvas_instructions

        async def event_generator():
            llm_service = get_llm_service()
            full_response = ""
            
            # Send user message ID first (remove _id for JSON serialization)
            user_msg_copy = {k: v for k, v in user_message.items() if k != '_id'}
            yield f"data: {json.dumps({'type': 'meta', 'user_message': user_msg_copy})}\n\n"

            try:
                async for chunk in llm_service.stream_chat(
                    messages=context_messages,
                    system_prompt=system_prompt,
                    model=request.model
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    
                # Save AI message after streaming
                ai_message_id = str(uuid.uuid4())
                ai_message = {
                    "id": ai_message_id,
                    "conversation_id": request.conversation_id,
                    "parent_id": user_message_id,
                    "content": full_response,
                    "role": "assistant",
                    "branch_name": request.branch_name,
                    "model_used": request.model,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                await db.messages.insert_one(ai_message)
                
                # Extract artifacts
                artifacts = extract_artifacts(full_response, request.conversation_id, ai_message_id)
                if artifacts:
                    await db.artifacts.insert_many(artifacts)
                
                # Auto-title
                msg_count = await db.messages.count_documents({"conversation_id": request.conversation_id})
                if msg_count <= 2:
                    title = request.content[:50] + ("..." if len(request.content) > 50 else "")
                    await db.conversations.update_one(
                        {"id": request.conversation_id},
                        {"$set": {"title": title}}
                    )


                # Remove _id for JSON serialization
                ai_msg_copy = {k: v for k, v in ai_message.items() if k != '_id'}
                yield f"data: {json.dumps({'type': 'done', 'ai_message': ai_msg_copy})}\n\n"
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat stream error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= TREE STRUCTURE ENDPOINT =============

@api_router.get("/conversations/{conversation_id}/tree")
async def get_conversation_tree(conversation_id: str):
    """Get full message tree structure for visualization"""
    messages = await db.messages.find({"conversation_id": conversation_id}, {"_id": 0}).to_list(10000)
    
    # Build tree structure
    nodes = []
    edges = []
    
    for msg in messages:
        nodes.append({
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"][:100],
            "branch_name": msg["branch_name"],
            "created_at": msg["created_at"]
        })
        if msg.get("parent_id"):
            edges.append({
                "from": msg["parent_id"],
                "to": msg["id"]
            })
    
    return {"nodes": nodes, "edges": edges}

# ============= FILE ENDPOINTS (RAG) =============

@api_router.post("/files/upload", status_code=201)
async def upload_file(file: UploadFile = File(...), workspace_id: str = Form(...)):
    """Upload a file to a workspace for RAG"""
    try:
        content = await file.read()
        text_content = content.decode('utf-8', errors='ignore')
        
        now = datetime.now(timezone.utc).isoformat()
        file_doc = {
            "id": str(uuid.uuid4()),
            "workspace_id": workspace_id,
            "filename": file.filename,
            "file_type": file.content_type or "text/plain",
            "content": text_content[:50000],  # Limit content size
            "created_at": now
        }
        await db.files.insert_one(file_doc)
        return FileDoc(**file_doc)
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/files", response_model=List[FileDoc])
async def get_files(workspace_id: str):
    """Get all files for a workspace"""
    files = await db.files.find({"workspace_id": workspace_id}, {"_id": 0}).to_list(100)
    return [FileDoc(**f) for f in files]

@api_router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    result = await db.files.delete_one({"id": file_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "File deleted"}

# ============= ARTIFACT ENDPOINTS =============

@api_router.get("/artifacts", response_model=List[Artifact])
async def get_artifacts(workspace_id: Optional[str] = None):
    """Get all artifacts, optionally filtered by workspace"""
    query = {}
    if workspace_id:
        # Get conversations in this workspace
        convs = await db.conversations.find({"workspace_id": workspace_id}, {"_id": 0, "id": 1}).to_list(1000)
        conv_ids = [c["id"] for c in convs]
        query["conversation_id"] = {"$in": conv_ids}
    
    artifacts = await db.artifacts.find(query, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return [Artifact(**a) for a in artifacts]

# ============= WORKSPACE INTELLIGENCE ENDPOINTS =============

@api_router.post("/workspaces/{workspace_id}/mindmap")
async def generate_mindmap(workspace_id: str, force_refresh: bool = False):
    """Generate or retrieve a mind map for the workspace"""
    try:
        # 1. Check cache (if not forcing refresh)
        if not force_refresh:
            existing = await db.visualizations.find_one(
                {"workspace_id": workspace_id, "type": "mindmap"},
                sort=[("created_at", -1)]
            )
            if existing:
                return {**existing["data"], "cached": True, "created_at": existing.get("created_at")}

        # 2. Get all messages
        convs = await db.conversations.find({"workspace_id": workspace_id}, {"_id": 0}).to_list(1000)
        if not convs:
            return {"nodes": [], "edges": [], "workspace_id": workspace_id, "message": "No conversations yet"}
        
        conv_ids = [c["id"] for c in convs]
        messages = await db.messages.find(
            {"conversation_id": {"$in": conv_ids}},
            {"_id": 0}
        ).to_list(10000)
        
        # Aggregate content
        all_content = "\n".join([m["content"] for m in messages])
        
        # 3. Generate with LLM
        llm_service = get_llm_service()
        prompt = f"Generate a comprehensive mind map from the following conversation content:\n\n{all_content[:12000]}\n\nReturn ONLY a JSON object with 'nodes' and 'links'. 'nodes' should have {{'id', 'name', 'val', 'color'}}. 'links' should have {{'source', 'target', 'label'}}. \nIMPORTANT: Every 'source' and 'target' in 'links' MUST match an 'id' in 'nodes'. Do not create links to non-existent nodes."
        
        response = await llm_service.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a mind map generator. Return strictly valid JSON. Nodes must have unique string IDs. Ensure referential integrity for all links.",
            model="gpt-4o-mini"
        )
        
        # 4. Parse & Save
        import json
        import re
        from datetime import datetime
        
        data = {"nodes": [], "edges": []} # Default
        try:
             json_match = re.search(r'\{[\s\S]*\}', response)
             if json_match:
                 parsed = json.loads(json_match.group())
                 # Normalize keys
                 data["nodes"] = parsed.get("nodes", [])
                 data["edges"] = parsed.get("links", []) if "links" in parsed else parsed.get("edges", [])
        except Exception as e:
             logger.error(f"JSON Parse Error: {e}")
             pass

        # Save to DB
        viz_doc = {
            "workspace_id": workspace_id,
            "type": "mindmap",
            "data": data,
            "created_at": datetime.utcnow()
        }
        await db.visualizations.insert_one(viz_doc)

        return {**data, "cached": False, "created_at": viz_doc["created_at"]}
        
    except Exception as e:
        logger.error(f"Mind map error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/workspaces/{workspace_id}/visualizations")
async def list_visualizations(workspace_id: str):
    """List all visualizations for a workspace for the gallery"""
    try:
        vizs = await db.visualizations.find(
            {"workspace_id": workspace_id},
            {"_id": 0, "data": 0} # Exclude heavy data, just metadata
        ).sort("created_at", -1).to_list(100)
        
        # Transform to fit ArtifactsGallery expected format (or close to it)
        # ArtifactsGallery expects {id, type, content, created_at}
        # We'll map 'mindmap'/'knowledge_graph' to a type usage.
        
        artifacts = []
        for v in vizs:
            artifacts.append({
                "id": str(v.get("_id", "")) if "_id" in v else "viz_" + str(int(v["created_at"].timestamp())),
                "type": "visualization_" + v["type"], # distinct type
                "content": f"Visualization: {v['type'].replace('_', ' ').title()}", # placeholder content description
                "created_at": v["created_at"],
                "viz_type": v["type"], # extra meta
                "workspace_id": v.get("workspace_id") # Add workspace context
            })
            
        return artifacts
    except Exception as e:
         logger.error(f"List viz error: {str(e)}")
         return []

@api_router.post("/workspaces/{workspace_id}/knowledge-graph")
async def generate_knowledge_graph(workspace_id: str, force_refresh: bool = False):
    """Generate or retrieve knowledge graph for the workspace"""
    try:
        # 1. Check cache
        if not force_refresh:
            existing = await db.visualizations.find_one(
                {"workspace_id": workspace_id, "type": "knowledge_graph"},
                sort=[("created_at", -1)]
            )
            if existing:
                return {**existing["data"], "cached": True, "created_at": existing.get("created_at")}

        # 2. Get Messages
        convs = await db.conversations.find({"workspace_id": workspace_id}, {"_id": 0}).to_list(1000)
        if not convs:
            return {"nodes": [], "edges": [], "workspace_id": workspace_id, "message": "No conversations yet"}
        
        conv_ids = [c["id"] for c in convs]
        messages = await db.messages.find(
            {"conversation_id": {"$in": conv_ids}},
            {"_id": 0}
        ).to_list(10000)
        
        all_content = "\n".join([m["content"] for m in messages])
        
        # 3. Generate
        llm_service = get_llm_service()
        prompt = f"Extract entities and relations from:\n\n{all_content[:12000]}\n\nReturn strictly valid JSON with keys: 'nodes' (list of {{'id', 'label', 'color', 'type'}}) and 'edges' (list of {{'from', 'to', 'relationship'}})."
        
        response = await llm_service.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are an entity extractor. Return strictly valid JSON. 'nodes' must include 'id' and 'label'. 'edges' must include 'from', 'to'. IMPORTANT: Ensure all node IDs are unique strings.",
            model="gpt-4o-mini"
        )
        
        # 4. Parse & Save
        import json
        from datetime import datetime
        
        data = {"nodes": [], "edges": []}
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                data["nodes"] = parsed.get("nodes", [])
                data["edges"] = parsed.get("edges", [])
        except Exception as e:
             logger.error(f"JSON Parse Error: {e}")
             pass
        
        # Save to DB
        viz_doc = {
            "workspace_id": workspace_id,
            "type": "knowledge_graph",
            "data": data,
            "created_at": datetime.utcnow()
        }
        await db.visualizations.insert_one(viz_doc)
        
        return {**data, "cached": False, "created_at": viz_doc["created_at"]}
        
    except Exception as e:
        logger.error(f"Knowledge graph error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= CANVAS ENDPOINTS =============

@api_router.post("/canvas")
async def create_canvas(input: CanvasCreate):
    """Create a new canvas artifact"""
    try:
        now = datetime.now(timezone.utc)
        
        canvas_doc = {
            "id": str(uuid.uuid4()),
            "content": input.content,
            "language": input.language,
            "workspace_id": input.workspace_id,
            "conversation_id": input.conversation_id,
            "versions": [{
                "id": str(uuid.uuid4()),
                "content": input.content,
                "language": input.language,
                "timestamp": now.isoformat(),
                "saved": True
            }],
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        await db.canvas_artifacts.insert_one(canvas_doc)
        
        return {
            "artifact_id": canvas_doc["id"],
            "version_id": canvas_doc["versions"][0]["id"],
            "created_at": canvas_doc["created_at"]
        }
    except Exception as e:
        logger.error(f"Create canvas error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/canvas/{canvas_id}")
async def get_canvas(canvas_id: str):
    """Get a canvas artifact by ID"""
    try:
        canvas = await db.canvas_artifacts.find_one(
            {"id": canvas_id},
            {"_id": 0}
        )
        
        if not canvas:
            raise HTTPException(status_code=404, detail="Canvas not found")
        
        return canvas
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get canvas error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/canvas/{canvas_id}/versions")
async def add_canvas_version(canvas_id: str, input: CanvasVersionCreate):
    """Add a new version to an existing canvas"""
    try:
        canvas = await db.canvas_artifacts.find_one({"id": canvas_id})
        
        if not canvas:
            raise HTTPException(status_code=404, detail="Canvas not found")
        
        now = datetime.now(timezone.utc)
        
        new_version = {
            "id": str(uuid.uuid4()),
            "content": input.content,
            "language": input.language or canvas.get("language", "html"),
            "timestamp": now.isoformat(),
            "saved": True
        }
        
        await db.canvas_artifacts.update_one(
            {"id": canvas_id},
            {
                "$push": {"versions": new_version},
                "$set": {
                    "content": input.content,
                    "language": input.language or canvas.get("language", "html"),
                    "updated_at": now.isoformat()
                }
            }
        )
        
        return {
            "artifact_id": canvas_id,
            "version_id": new_version["id"],
            "version_count": len(canvas.get("versions", [])) + 1
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add canvas version error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/canvas/{canvas_id}/versions")
async def list_canvas_versions(canvas_id: str):
    """List all versions of a canvas"""
    try:
        canvas = await db.canvas_artifacts.find_one(
            {"id": canvas_id},
            {"_id": 0, "versions": 1}
        )
        
        if not canvas:
            raise HTTPException(status_code=404, detail="Canvas not found")
        
        return canvas.get("versions", [])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List canvas versions error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.delete("/canvas/{canvas_id}")
async def delete_canvas(canvas_id: str):
    """Delete a canvas artifact"""
    try:
        result = await db.canvas_artifacts.delete_one({"id": canvas_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Canvas not found")
        
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete canvas error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= HEALTH & ROOT =============

@api_router.get("/")
async def root():
    return {"message": "SynapsBranch API"}

@api_router.get("/health")
async def health():
    return {"status": "healthy"}

# CORS MUST be added BEFORE including the router
# This ensures preflight OPTIONS requests are handled correctly
# Note: allow_credentials=True does NOT work with allow_origins=["*"]
# You must specify exact origins when using credentials
_cors_allow_all = settings.CORS_ORIGINS == "*"
logger.info(f"CORS Configuration: origins={settings.cors_origins_list}, allow_all={_cors_allow_all}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=not _cors_allow_all,  # Credentials only with specific origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include router AFTER CORS middleware
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

