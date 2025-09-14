"""
Authentication module for AirWatch Pro
"""

import streamlit as st
import hashlib
import json
import os
from datetime import datetime

class AuthManager:
    def __init__(self, users_file='users.json'):
        self.users_file = users_file
        self.ensure_users_file()
    
    def ensure_users_file(self):
        """Ensure users file exists"""
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self):
        """Load users from JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_users(self, users):
        """Save users to JSON file"""
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def register_user(self, username, password, email):
        """Register a new user"""
        users = self.load_users()
        
        if username in users:
            return False, "Username already exists"
        
        if not username or not password or not email:
            return False, "All fields are required"
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        users[username] = {
            'password': self.hash_password(password),
            'email': email,
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'login_count': 0
        }
        
        self.save_users(users)
        return True, "User registered successfully"
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials"""
        users = self.load_users()
        
        if username not in users:
            return False, "Username not found"
        
        if users[username]['password'] != self.hash_password(password):
            return False, "Invalid password"
        
        # Update login info
        users[username]['last_login'] = datetime.now().isoformat()
        users[username]['login_count'] = users[username].get('login_count', 0) + 1
        self.save_users(users)
        
        return True, "Login successful"
    
    def get_user_info(self, username):
        """Get user information"""
        users = self.load_users()
        return users.get(username, {})
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        users = self.load_users()
        
        if username not in users:
            return False, "User not found"
        
        if users[username]['password'] != self.hash_password(old_password):
            return False, "Current password is incorrect"
        
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters long"
        
        users[username]['password'] = self.hash_password(new_password)
        self.save_users(users)
        
        return True, "Password changed successfully"
