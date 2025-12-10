"""
Settings module using Pydantic Settings for configuration management.
Prioritizes environment variables (production) over .env file (local development).
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    In production (Railway), environment variables are set directly.
    In local development, values are loaded from .env file.
    
    Critical settings (MONGO_URL, JWT_SECRET_KEY) have no defaults
    and will cause the app to crash if not set - this is intentional.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",  # Ignore extra env vars not defined here
    )
    
    # ===========================================
    # DATABASE - REQUIRED (no defaults)
    # ===========================================
    MONGO_URL: str = Field(..., description="MongoDB connection string")
    DB_NAME: str = Field(default="synapsbranch", description="Database name")
    
    # ===========================================
    # JWT AUTHENTICATION - REQUIRED (no defaults for secret)
    # ===========================================
    JWT_SECRET_KEY: str = Field(..., description="JWT secret key - MUST be set in production")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440, description="Token expiry in minutes (24h default)")
    
    # ===========================================
    # OAUTH - OPTIONAL (empty string = disabled)
    # ===========================================
    GOOGLE_CLIENT_ID: str = Field(default="", description="Google OAuth Client ID")
    GOOGLE_CLIENT_SECRET: str = Field(default="", description="Google OAuth Client Secret")
    GITHUB_CLIENT_ID: str = Field(default="", description="GitHub OAuth Client ID")
    GITHUB_CLIENT_SECRET: str = Field(default="", description="GitHub OAuth Client Secret")
    
    # ===========================================
    # FRONTEND/CORS CONFIGURATION
    # ===========================================
    FRONTEND_URL: str = Field(default="http://localhost:3000", description="Frontend URL for OAuth redirects")
    CORS_ORIGINS: str = Field(default="*", description="Comma-separated list of allowed origins")
    
    # ===========================================
    # AWS BEDROCK - OPTIONAL (for LLM)
    # ===========================================
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, description="AWS Access Key ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, description="AWS Secret Access Key")
    AWS_BEDROCK_REGION: str = Field(default="us-east-1", description="AWS Bedrock region")
    
    # ===========================================
    # OPENAI - OPTIONAL (fallback LLM)
    # ===========================================
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API Key")
    OPENAI_ORG_ID: Optional[str] = Field(default=None, description="OpenAI Organization ID")
    
    # ===========================================
    # LLM CONFIGURATION
    # ===========================================
    DEFAULT_LLM_MODEL: str = Field(default="amazon.nova-pro-v1:0", description="Default LLM model")
    
    # ===========================================
    # COMPUTED PROPERTIES
    # ===========================================
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS string into a list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def google_oauth_enabled(self) -> bool:
        """Check if Google OAuth is configured."""
        return bool(self.GOOGLE_CLIENT_ID and self.GOOGLE_CLIENT_SECRET)
    
    @property
    def github_oauth_enabled(self) -> bool:
        """Check if GitHub OAuth is configured."""
        return bool(self.GITHUB_CLIENT_ID and self.GITHUB_CLIENT_SECRET)
    
    @property
    def aws_configured(self) -> bool:
        """Check if AWS credentials are configured."""
        return bool(self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY)
    
    @property
    def openai_configured(self) -> bool:
        """Check if OpenAI is configured."""
        return bool(self.OPENAI_API_KEY)


# Singleton instance - created on import
# This will raise ValidationError if required fields are missing
settings = Settings()
