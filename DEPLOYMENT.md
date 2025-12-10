# SynapsBranch Backend - Railway Deployment Guide

## 🚀 Déploiement sur Railway

### Prérequis

1. Compte Railway: https://railway.app
2. MongoDB Atlas (gratuit): https://www.mongodb.com/cloud/atlas
3. Codes d'accès OAuth (Google/GitHub) - optionnel

---

## 📋 Étapes de Déploiement

### 1. Créer un Projet sur Railway

```bash
# Installer Railway CLI (optionnel)
npm i -g @railway/cli

# Login
railway login
```

Ou utilisez l'interface web: https://railway.app/new

### 2. Configurer MongoDB Atlas

1. Créez un cluster gratuit sur MongoDB Atlas
2. Créez une base de données `synapsbranch`
3. Configurez l'accès réseau: **Allow Access from Anywhere** (0.0.0.0/0)
4. Créez un utilisateur de base de données
5. Copiez la connection string

**Format:**
```
mongodb+srv://username:password@cluster.mongodb.net/synapsbranch?retryWrites=true&w=majority
```

### 3. Déployer sur Railway

#### Option A: Via l'Interface Web

1. Allez sur https://railway.app/new
2. Sélectionnez "Deploy from GitHub repo"
3. Connectez votre repository
4. Sélectionnez le dossier `backend` comme root directory
5. Railway détectera automatiquement Python et FastAPI

#### Option B: Via CLI

```bash
cd backend
railway init
railway up
```

### 4. Configurer les Variables d'Environnement

Dans Railway Dashboard → Variables:

```env
# MongoDB (REQUIS)
MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/synapsbranch
DB_NAME=synapsbranch

# JWT Secret (REQUIS - générer avec generate_jwt_secret.py)
JWT_SECRET_KEY=votre-cle-secrete-complexe-de-64-chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Frontend URL (REQUIS - sera l'URL de votre frontend déployé)
FRONTEND_URL=https://votre-frontend.vercel.app

# CORS Origins (REQUIS)
CORS_ORIGINS=https://votre-frontend.vercel.app

# OAuth Google (OPTIONNEL)
GOOGLE_CLIENT_ID=votre-google-client-id
GOOGLE_CLIENT_SECRET=votre-google-secret

# OAuth GitHub (OPTIONNEL)
GITHUB_CLIENT_ID=votre-github-client-id
GITHUB_CLIENT_SECRET=votre-github-secret

# LLM API Keys (si utilisé)
OPENAI_API_KEY=votre-openai-key
```

### 5. Générer JWT Secret (LOCAL UNIQUEMENT)

```bash
# Sur votre machine locale
python generate_jwt_secret.py

# Copiez la clé de 64 caractères dans Railway
```

### 6. Générer les Codes d'Invitation (LOCAL UNIQUEMENT)

```bash
# Sur votre machine locale avec MONGO_URL de production
python generate_invite_codes.py --count 10

# Les codes seront insérés directement dans votre base MongoDB Atlas
```

**⚠️ IMPORTANT:** Ne JAMAIS déployer `generate_invite_codes.py` ou `generate_jwt_secret.py` en production!

### 7. Vérifier le Déploiement

Une fois déployé, Railway vous donnera une URL:
```
https://votre-backend.up.railway.app
```

Testez:
```bash
curl https://votre-backend.up.railway.app/api/health

# Devrait retourner: {"status":"healthy"}
```

---

## 🔒 Configuration OAuth pour Production

### Google Cloud Console

**Authorized redirect URIs:**
```
https://votre-frontend.vercel.app/auth/callback
```

### GitHub Developer Settings

**Authorization callback URL:**
```
https://votre-frontend.vercel.app/auth/callback
```

---

## 📊 Monitoring

Railway fournit automatiquement:
- Logs en temps réel
- Métriques CPU/RAM
- Health checks
- Auto-restart en cas d'erreur

Accédez aux logs: **Railway Dashboard → Deployments → View Logs**

---

## 🔧 Commandes Utiles

```bash
# Voir les logs
railway logs

# Redéployer
railway up

# Ouvrir le dashboard
railway open

# Voir les variables d'environnement
railway variables
```

---

## 🚨 Troubleshooting

### Erreur: "Application failed to respond"

- Vérifiez que `PORT` est bien utilisé: `--port $PORT`
- Vérifiez `MONGO_URL` dans les variables d'environnement
- Consultez les logs: `railway logs`

### Erreur: "MongoDB connection failed"

- Vérifiez la connection string MongoDB Atlas
- Assurez-vous que l'IP 0.0.0.0/0 est autorisée sur Atlas
- Vérifiez username/password

### OAuth ne fonctionne pas

- Vérifiez que `FRONTEND_URL` pointe vers votre frontend déployé
- Vérifiez les callback URLs dans Google/GitHub
- Assurez-vous que `CORS_ORIGINS` inclut votre frontend

---

## 📝 Checklist Avant Déploiement

- [ ] MongoDB Atlas configuré avec connection string
- [ ] JWT secret généré (64+ caractères)
- [ ] Variables d'environnement configurées dans Railway
- [ ] `generate_invite_codes.py` et `generate_jwt_secret.py` dans `.gitignore`
- [ ] Codes d'invitation générés en local et insérés dans MongoDB
- [ ] OAuth configuré avec les URLs de production
- [ ] `FRONTEND_URL` pointe vers le frontend déployé
- [ ] `CORS_ORIGINS` inclut le frontend déployé
- [ ] Health check testé: `/api/health`

---

## 🌐 URLs Importantes

| Service | URL |
|---------|-----|
| Backend API | `https://votre-backend.up.railway.app/api` |
| API Docs | `https://votre-backend.up.railway.app/docs` |
| Health Check | `https://votre-backend.up.railway.app/api/health` |
| MongoDB Atlas | https://cloud.mongodb.com |
| Railway Dashboard | https://railway.app/dashboard |

---

## 💡 Production Best Practices

1. ✅ Utilisez MongoDB Atlas (ne pas utiliser MongoDB local)
2. ✅ Générez un JWT secret fort (64+ chars)
3. ✅ Configurez CORS avec votre domaine exact
4. ✅ Ne commitez JAMAIS les scripts de génération de codes
5. ✅ Utilisez des codes d'invitation sécurisés (format SB-XXXX-XXXX-XXXX)
6. ✅ Activez le monitoring dans Railway
7. ✅ Gardez une sauvegarde de vos codes d'invitation non utilisés

---

## 🆘 Support

En cas de problème:
1. Consultez les logs Railway
2. Vérifiez les variables d'environnement
3. Testez la connexion MongoDB Atlas
4. Vérifiez le health check endpoint

**Bon déploiement! 🚀**
