# Authentication and Authorization

## Overview

The Cloud Migration Advisor API uses JWT (JSON Web Tokens) for authentication and role-based access control (RBAC) for authorization. All API endpoints require valid authentication tokens.

## Authentication Flow

### 1. User Login

Users authenticate by providing credentials to obtain a JWT token:

**Endpoint**: `POST /auth/login`

**Request**:
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNjk5ODc2ODAwfQ.signature",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here"
}
```

### 2. Token Usage

Include the access token in the Authorization header for all API requests:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 3. Token Refresh

When the access token expires, use the refresh token to obtain a new one:

**Endpoint**: `POST /auth/refresh`

**Request**:
```json
{
  "refresh_token": "refresh_token_here"
}
```

**Response**:
```json
{
  "access_token": "new_access_token",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 4. Logout

Invalidate tokens when logging out:

**Endpoint**: `POST /auth/logout`

**Headers**:
```
Authorization: Bearer your_access_token
```

## JWT Token Structure

The JWT token contains the following claims:

```json
{
  "sub": "user@example.com",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "roles": ["migration_admin", "viewer"],
  "organization_id": "org-123",
  "exp": 1699876800,
  "iat": 1699873200
}
```

**Claims**:
- `sub`: Subject (username/email)
- `user_id`: Unique user identifier
- `roles`: Array of user roles
- `organization_id`: Organization identifier
- `exp`: Expiration timestamp
- `iat`: Issued at timestamp

## Authorization Roles

The system implements role-based access control with the following roles:

### 1. Migration Admin

**Permissions**:
- Create, read, update, delete migration projects
- Manage organization profiles
- Generate and modify recommendations
- Create and execute migration plans
- Discover and organize resources
- Configure FinOps integration
- Generate reports
- Manage users and permissions

**Use Case**: Project managers and migration leads

### 2. Migration Engineer

**Permissions**:
- Read migration projects
- Create and update workload profiles
- Create and update requirements
- View recommendations
- Execute migration phases
- Discover and categorize resources
- View reports

**Use Case**: Engineers executing migrations

### 3. Analyst

**Permissions**:
- Read migration projects
- View organization profiles
- View workload profiles and requirements
- View recommendations and comparisons
- View migration plans
- View resource inventory
- Generate and view reports

**Use Case**: Business analysts and stakeholders

### 4. Viewer

**Permissions**:
- Read-only access to migration projects
- View organization profiles
- View recommendations
- View migration progress
- View resource inventory
- View reports

**Use Case**: Executives and observers

## Role-Based Endpoint Access

| Endpoint | Migration Admin | Migration Engineer | Analyst | Viewer |
|----------|----------------|-------------------|---------|--------|
| POST /api/migrations/projects | ✓ | ✗ | ✗ | ✗ |
| GET /api/migrations/projects | ✓ | ✓ | ✓ | ✓ |
| DELETE /api/migrations/projects/{id} | ✓ | ✗ | ✗ | ✗ |
| POST /api/migrations/{id}/assessment/organization | ✓ | ✓ | ✗ | ✗ |
| POST /api/migrations/{id}/workloads | ✓ | ✓ | ✗ | ✗ |
| POST /api/migrations/{id}/recommendations/generate | ✓ | ✓ | ✗ | ✗ |
| GET /api/migrations/{id}/recommendations | ✓ | ✓ | ✓ | ✓ |
| POST /api/migrations/{id}/plan | ✓ | ✓ | ✗ | ✗ |
| PUT /api/migrations/{id}/plan/phases/{phase_id}/status | ✓ | ✓ | ✗ | ✗ |
| GET /api/migrations/{id}/plan/progress | ✓ | ✓ | ✓ | ✓ |
| POST /api/migrations/{id}/resources/discover | ✓ | ✓ | ✗ | ✗ |
| GET /api/migrations/{id}/resources | ✓ | ✓ | ✓ | ✓ |
| POST /api/migrations/{id}/integration/finops | ✓ | ✗ | ✗ | ✗ |
| GET /api/migrations/{id}/reports/final | ✓ | ✓ | ✓ | ✓ |

## Resource-Level Authorization

In addition to role-based access, the system implements resource-level authorization:

### Organization Isolation

Users can only access migration projects within their organization:

```python
# User can only see projects from their organization
GET /api/migrations/projects
# Returns only projects where project.organization_id == user.organization_id
```

### Project Ownership

Users can be assigned as project owners or team members:

```python
# Check if user has access to project
if user.role == 'migration_admin' or user.id in project.team_members:
    # Allow access
else:
    # Deny access (403 Forbidden)
```

## Security Best Practices

### 1. Token Storage

**Client-Side**:
- Store tokens in memory or secure storage (e.g., HttpOnly cookies)
- Never store tokens in localStorage for production applications
- Clear tokens on logout

**Example (JavaScript)**:
```javascript
// Store token securely
sessionStorage.setItem('access_token', token);

// Retrieve token
const token = sessionStorage.getItem('access_token');

// Clear on logout
sessionStorage.removeItem('access_token');
```

### 2. Token Expiration

- Access tokens expire after 1 hour
- Refresh tokens expire after 7 days
- Implement automatic token refresh before expiration

**Example (Python)**:
```python
import time
import jwt

def is_token_expired(token):
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get('exp')
        return time.time() > exp
    except:
        return True

def refresh_if_needed(access_token, refresh_token):
    if is_token_expired(access_token):
        # Refresh token
        response = requests.post(
            'https://api.cloudmigration.example.com/v1/auth/refresh',
            json={'refresh_token': refresh_token}
        )
        return response.json()['access_token']
    return access_token
```

### 3. HTTPS Only

- All API requests must use HTTPS
- HTTP requests will be rejected

### 4. Rate Limiting

- Failed authentication attempts are rate-limited
- Maximum 5 failed attempts per 15 minutes
- Account lockout after 10 failed attempts

### 5. Password Requirements

- Minimum 12 characters
- Must include uppercase, lowercase, numbers, and special characters
- Cannot reuse last 5 passwords
- Must change password every 90 days

### 6. Multi-Factor Authentication (MFA)

MFA is required for Migration Admin roles:

**Enable MFA**:
```bash
POST /auth/mfa/enable
Authorization: Bearer your_token

Response:
{
  "qr_code": "data:image/png;base64,...",
  "secret": "JBSWY3DPEHPK3PXP",
  "backup_codes": ["12345678", "87654321", ...]
}
```

**Verify MFA**:
```bash
POST /auth/mfa/verify
{
  "code": "123456"
}
```

**Login with MFA**:
```bash
POST /auth/login
{
  "username": "user@example.com",
  "password": "secure_password",
  "mfa_code": "123456"
}
```

## API Key Authentication

For programmatic access (CI/CD, automation), use API keys:

### Generate API Key

**Endpoint**: `POST /auth/api-keys`

**Request**:
```json
{
  "name": "CI/CD Pipeline",
  "expires_in_days": 90,
  "scopes": ["read:projects", "write:resources"]
}
```

**Response**:
```json
{
  "api_key": "ma_live_1234567890abcdef",
  "name": "CI/CD Pipeline",
  "created_at": "2023-11-16T12:00:00Z",
  "expires_at": "2024-02-14T12:00:00Z",
  "scopes": ["read:projects", "write:resources"]
}
```

### Use API Key

Include the API key in the Authorization header:

```
Authorization: ApiKey ma_live_1234567890abcdef
```

### API Key Scopes

Available scopes:
- `read:projects` - Read migration projects
- `write:projects` - Create and update projects
- `read:resources` - Read resource inventory
- `write:resources` - Discover and organize resources
- `read:reports` - Generate and view reports
- `admin:all` - Full administrative access

## Error Responses

### 401 Unauthorized

Missing or invalid authentication token:

```json
{
  "detail": "Could not validate credentials"
}
```

### 403 Forbidden

Insufficient permissions:

```json
{
  "detail": "User does not have permission to perform this action"
}
```

### 429 Too Many Requests

Rate limit exceeded:

```json
{
  "detail": "Too many authentication attempts. Please try again in 15 minutes."
}
```

## Implementation Examples

### Python Example

```python
import requests
from datetime import datetime, timedelta

class MigrationAdvisorAuth:
    def __init__(self, api_url, username, password):
        self.api_url = api_url
        self.username = username
        self.password = password
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = None
    
    def login(self):
        response = requests.post(
            f'{self.api_url}/auth/login',
            json={
                'username': self.username,
                'password': self.password
            }
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data['access_token']
        self.refresh_token = data['refresh_token']
        self.token_expiry = datetime.now() + timedelta(seconds=data['expires_in'])
        
        return self.access_token
    
    def get_token(self):
        if not self.access_token or datetime.now() >= self.token_expiry:
            if self.refresh_token:
                self.refresh()
            else:
                self.login()
        return self.access_token
    
    def refresh(self):
        response = requests.post(
            f'{self.api_url}/auth/refresh',
            json={'refresh_token': self.refresh_token}
        )
        response.raise_for_status()
        
        data = response.json()
        self.access_token = data['access_token']
        self.token_expiry = datetime.now() + timedelta(seconds=data['expires_in'])
        
        return self.access_token
    
    def get_headers(self):
        return {
            'Authorization': f'Bearer {self.get_token()}',
            'Content-Type': 'application/json'
        }

# Usage
auth = MigrationAdvisorAuth(
    api_url='https://api.cloudmigration.example.com/v1',
    username='user@example.com',
    password='secure_password'
)

# Make authenticated request
response = requests.get(
    'https://api.cloudmigration.example.com/v1/api/migrations/projects',
    headers=auth.get_headers()
)
```

### JavaScript Example

```javascript
class MigrationAdvisorAuth {
  constructor(apiUrl, username, password) {
    this.apiUrl = apiUrl;
    this.username = username;
    this.password = password;
    this.accessToken = null;
    this.refreshToken = null;
    this.tokenExpiry = null;
  }
  
  async login() {
    const response = await fetch(`${this.apiUrl}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: this.username,
        password: this.password
      })
    });
    
    if (!response.ok) {
      throw new Error('Authentication failed');
    }
    
    const data = await response.json();
    this.accessToken = data.access_token;
    this.refreshToken = data.refresh_token;
    this.tokenExpiry = Date.now() + (data.expires_in * 1000);
    
    return this.accessToken;
  }
  
  async getToken() {
    if (!this.accessToken || Date.now() >= this.tokenExpiry) {
      if (this.refreshToken) {
        await this.refresh();
      } else {
        await this.login();
      }
    }
    return this.accessToken;
  }
  
  async refresh() {
    const response = await fetch(`${this.apiUrl}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        refresh_token: this.refreshToken
      })
    });
    
    if (!response.ok) {
      // Refresh failed, need to login again
      return await this.login();
    }
    
    const data = await response.json();
    this.accessToken = data.access_token;
    this.tokenExpiry = Date.now() + (data.expires_in * 1000);
    
    return this.accessToken;
  }
  
  async getHeaders() {
    const token = await this.getToken();
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }
}

// Usage
const auth = new MigrationAdvisorAuth(
  'https://api.cloudmigration.example.com/v1',
  'user@example.com',
  'secure_password'
);

// Make authenticated request
const headers = await auth.getHeaders();
const response = await fetch(
  'https://api.cloudmigration.example.com/v1/api/migrations/projects',
  { headers }
);
```

## Troubleshooting

### Token Expired

**Problem**: Receiving 401 errors with "Token expired" message

**Solution**: Implement automatic token refresh or re-authenticate

### Invalid Token

**Problem**: Receiving 401 errors with "Invalid token" message

**Solutions**:
- Verify token is correctly formatted
- Check token hasn't been revoked
- Ensure token is included in Authorization header
- Verify using correct token type (Bearer vs ApiKey)

### Insufficient Permissions

**Problem**: Receiving 403 errors

**Solutions**:
- Verify user has correct role assigned
- Check user is member of project team
- Verify organization access
- Contact administrator to adjust permissions

## Security Contacts

For security issues or vulnerabilities:

- **Security Email**: security@cloudmigration.example.com
- **Bug Bounty**: https://cloudmigration.example.com/security/bounty
- **PGP Key**: Available at https://cloudmigration.example.com/security/pgp
