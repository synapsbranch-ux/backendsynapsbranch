# 🚀 Checklist de Déploiement Railway - SynapsBranch Backend

## Avant le Déploiement

### 1. MongoDB Atlas
- [ ] Compte MongoDB Atlas créé
- [ ] Cluster gratuit créé (M0)
- [ ] Base de données `synapsbranch` créée
- [ ] Accès réseau configuré (0.0.0.0/0)
- [ ] Utilisateur de base de données créé
- [ ] Connection string copiée

### 2. Sécurité
- [ ] JWT secret généré localement (`python generate_jwt_secret.py`)
- [ ] Codes d'invitation générés localement (`python generate_invite_codes.py --count 10`)
- [ ] Scripts sensibles dans `.gitignore` (generate_*.py)
- [ ] Fichier `.env` dans `.gitignore`

### 3. OAuth (Optionnel)
- [ ] Google OAuth App créée
- [ ] GitHub OAuth App créée  
- [ ] Client IDs et Secrets copiés

---

## Déploiement Railway

### 4. Configuration Railway
- [ ] Projet Railway créé
- [ ] Repository GitHub connecté
- [ ] Root directory: `backend` configuré
- [ ] Build configuré (automatique avec `railway.json`)

### 5. Variables d'Environnement Railway

**Variables REQUISES:**
```
MONGO_URL=mongodb+srv://...
DB_NAME=synapsbranch
JWT_SECRET_KEY=<64-char-secret>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=1440
FRONTEND_URL=https://votre-frontend.vercel.app
CORS_ORIGINS=https://votre-frontend.vercel.app
```

**Variables OPTIONNELLES (OAuth):**
```
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
```

**Variables OPTIONNELLES (LLM):**
```
OPENAI_API_KEY=...
```

- [ ] Toutes les variables requises ajoutées
- [ ] URLs de frontend mises à jour pour production
- [ ] Secrets sécurisés copiés correctement

### 6. Configuration OAuth Production

**Google Cloud Console:**
- [ ] Authorized redirect URI: `https://votre-frontend.vercel.app/auth/callback`

**GitHub Developer Settings:**
- [ ] Callback URL: `https://votre-frontend.vercel.app/auth/callback`

---

## Vérification Post-Déploiement

### 7. Tests de Base
```bash
# Health check
curl https://votre-backend.up.railway.app/api/health

# Docs API
https://votre-backend.up.railway.app/docs
```

- [ ] Health endpoint répond avec `{"status":"healthy"}`
- [ ] API docs accessibles
- [ ] Logs Railway sans erreurs critiques

### 8. Tests Fonctionnels
- [ ] Inscription email fonctionne
- [ ] Login email fonctionne
- [ ] Google OAuth fonctionne (si configuré)
- [ ] GitHub OAuth fonctionne (si configuré)
- [ ] Page invite code fonctionne
- [ ] Validation code d'invitation fonctionne
- [ ] Logout fonctionne

---

## Maintenance

### 9. Codes d'Invitation
- [ ] Codes générés et sauvegardés localement
- [ ] Fichier `invite_codes.txt` gardé en sécurité
- [ ] Log de qui reçoit quel code (pour support)

### 10. Monitoring
- [ ] Logs Railway configurés
- [ ] Alertes Rails (optionnel) configurées
- [ ] Backup MongoDB Atlas configuré

---

## Commandes Utiles

```bash
# Logs en temps réel
railway logs

# Redéployer
railway up

# Variables d'environnement
railway variables
```

---

## URLs de Référence

| Service | URL |
|---------|-----|
| Backend API | https://votre-backend.up.railway.app/api |
| API Docs | https://votre-backend.up.railway.app/docs |
| Railway Dashboard | https://railway.app/dashboard |
| MongoDB Atlas | https://cloud.mongodb.com |

---

## 🆘 En Cas de Problème

1. ✅ Vérifier les logs: `railway logs`
2. ✅ Vérifier les variables d'environnement
3. ✅ Tester connection MongoDB: ping depuis Atlas
4. ✅ Vérifier CORS et FRONTEND_URL
5. ✅ Consulter `DEPLOYMENT.md` pour troubleshooting détaillé

---

**Date de déploiement:** __________
**URL Backend:** __________
**URL Frontend:** __________
**MongoDB Cluster:** __________
