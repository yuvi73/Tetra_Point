#!/usr/bin/env python3
"""
Tetralemma Database System
==========================

A demonstration of how Tetralemma Space (𝕋) logic can be applied to database systems
for handling contradictions, missing data, and indeterminacy.
"""

import sqlite3
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

class TetralemmaValue(Enum):
    """Four-valued logic for database operations"""
    EXPRESSED = 1      # Definitely true/known
    SUPPRESSED = 0     # Definitely false/known
    INAPPLICABLE = -1  # Neither true nor false (unknown/missing)
    EMPTY = -2         # No conceptual ground (contradiction resolved)

@dataclass
class Tetrapoint:
    """A tetrapoint representing the four logical positions"""
    a: TetralemmaValue          # Affirmation
    not_a: TetralemmaValue      # Negation
    both: TetralemmaValue       # Both true and false
    neither: TetralemmaValue    # Neither true nor false
    
    def __str__(self):
        return f"({self.a.name}, {self.not_a.name}, {self.both.name}, {self.neither.name})"
    
    def to_dict(self):
        return {
            'a': self.a.name,
            'not_a': self.not_a.name,
            'both': self.both.name,
            'neither': self.neither.name
        }

class TetralemmaDatabase:
    """A database system that uses Tetralemma logic for handling contradictions and uncertainty"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
    
    def setup_database(self):
        """Initialize the database with Tetralemma-aware tables"""
        cursor = self.conn.cursor()
        
        # Create a table for user status with Tetralemma values
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                is_active_tetrapoint TEXT,  -- JSON representation of tetrapoint
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create a table for data sources to track provenance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY,
                source_name TEXT,
                reliability_score REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create a table for contradiction tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY,
                entity_id INTEGER,
                field_name TEXT,
                source1_value TEXT,
                source2_value TEXT,
                tetrapoint_result TEXT,  -- JSON representation
                resolution_status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def insert_user(self, name: str, is_active_tetrapoint: Tetrapoint, email: str = None):
        """Insert a user with Tetralemma logic for active status"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO users (name, is_active_tetrapoint, email)
            VALUES (?, ?, ?)
        ''', (name, json.dumps(is_active_tetrapoint.to_dict()), email))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_data_source(self, source_name: str, reliability_score: float):
        """Insert a data source with reliability score"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO data_sources (source_name, reliability_score)
            VALUES (?, ?)
        ''', (source_name, reliability_score))
        self.conn.commit()
        return cursor.lastrowid
    
    def record_contradiction(self, entity_id: int, field_name: str, 
                           source1_value: str, source2_value: str, 
                           tetrapoint_result: Tetrapoint):
        """Record a contradiction between data sources"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO contradictions (entity_id, field_name, source1_value, 
                                      source2_value, tetrapoint_result, resolution_status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entity_id, field_name, source1_value, source2_value, 
              json.dumps(tetrapoint_result.to_dict()), 'unresolved'))
        self.conn.commit()
    
    def tetralemma_query(self, query_type: str = "all") -> pd.DataFrame:
        """Query users with Tetralemma logic interpretation"""
        if query_type == "all":
            query = "SELECT * FROM users"
        elif query_type == "active":
            # Query for users where affirmation is EXPRESSED
            query = '''
                SELECT * FROM users 
                WHERE json_extract(is_active_tetrapoint, '$.a') = 'EXPRESSED'
            '''
        elif query_type == "inactive":
            # Query for users where negation is EXPRESSED
            query = '''
                SELECT * FROM users 
                WHERE json_extract(is_active_tetrapoint, '$.not_a') = 'EXPRESSED'
            '''
        elif query_type == "contradictory":
            # Query for users where both is EXPRESSED (contradiction)
            query = '''
                SELECT * FROM users 
                WHERE json_extract(is_active_tetrapoint, '$.both') = 'EXPRESSED'
            '''
        elif query_type == "unknown":
            # Query for users where neither is EXPRESSED (unknown)
            query = '''
                SELECT * FROM users 
                WHERE json_extract(is_active_tetrapoint, '$.neither') = 'EXPRESSED'
            '''
        else:
            query = "SELECT * FROM users"
        
        return pd.read_sql_query(query, self.conn)
    
    def get_contradictions_summary(self) -> pd.DataFrame:
        """Get a summary of all contradictions in the database"""
        query = '''
            SELECT 
                field_name,
                COUNT(*) as contradiction_count,
                GROUP_CONCAT(DISTINCT resolution_status) as statuses
            FROM contradictions 
            GROUP BY field_name
        '''
        return pd.read_sql_query(query, self.conn)
    
    def resolve_contradiction(self, contradiction_id: int, resolution: str):
        """Mark a contradiction as resolved"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE contradictions 
            SET resolution_status = ? 
            WHERE id = ?
        ''', (resolution, contradiction_id))
        self.conn.commit()

def demonstrate_tetralemma_database():
    """Demonstrate the Tetralemma database system with real-world scenarios"""
    
    print("🧠 Tetralemma Database System Demonstration")
    print("=" * 50)
    
    # Initialize database
    db = TetralemmaDatabase()
    
    # Insert some data sources
    db.insert_data_source("HR System", 0.9)
    db.insert_data_source("Active Directory", 0.8)
    db.insert_data_source("User Survey", 0.7)
    db.insert_data_source("Legacy System", 0.6)
    
    print("\n📊 Inserting users with Tetralemma logic:")
    
    # User 1: Definitely active (from HR system)
    user1_tetrapoint = Tetrapoint(
        a=TetralemmaValue.EXPRESSED,      # Definitely active
        not_a=TetralemmaValue.SUPPRESSED, # Not inactive
        both=TetralemmaValue.SUPPRESSED,  # No contradiction
        neither=TetralemmaValue.SUPPRESSED # Not unknown
    )
    db.insert_user("Alice Johnson", user1_tetrapoint, "alice@company.com")
    print(f"Alice: {user1_tetrapoint}")
    
    # User 2: Definitely inactive (from HR system)
    user2_tetrapoint = Tetrapoint(
        a=TetralemmaValue.SUPPRESSED,     # Not active
        not_a=TetralemmaValue.EXPRESSED,  # Definitely inactive
        both=TetralemmaValue.SUPPRESSED,  # No contradiction
        neither=TetralemmaValue.SUPPRESSED # Not unknown
    )
    db.insert_user("Bob Smith", user2_tetrapoint, "bob@company.com")
    print(f"Bob: {user2_tetrapoint}")
    
    # User 3: Contradictory information (HR says active, AD says inactive)
    user3_tetrapoint = Tetrapoint(
        a=TetralemmaValue.EXPRESSED,      # HR says active
        not_a=TetralemmaValue.EXPRESSED,  # AD says inactive
        both=TetralemmaValue.EXPRESSED,   # Contradiction present
        neither=TetralemmaValue.SUPPRESSED # Not unknown
    )
    db.insert_user("Carol Davis", user3_tetrapoint, "carol@company.com")
    print(f"Carol: {user3_tetrapoint}")
    
    # Record the contradiction
    db.record_contradiction(3, "is_active", "active", "inactive", user3_tetrapoint)
    
    # User 4: Unknown status (no data from any source)
    user4_tetrapoint = Tetrapoint(
        a=TetralemmaValue.SUPPRESSED,     # Not known to be active
        not_a=TetralemmaValue.SUPPRESSED, # Not known to be inactive
        both=TetralemmaValue.SUPPRESSED,  # No contradiction
        neither=TetralemmaValue.EXPRESSED # Unknown status
    )
    db.insert_user("David Wilson", user4_tetrapoint, "david@company.com")
    print(f"David: {user4_tetrapoint}")
    
    # User 5: Empty state (contradiction resolved to emptiness)
    user5_tetrapoint = Tetrapoint(
        a=TetralemmaValue.EMPTY,          # All positions empty
        not_a=TetralemmaValue.EMPTY,
        both=TetralemmaValue.EMPTY,
        neither=TetralemmaValue.EMPTY
    )
    db.insert_user("Eve Brown", user5_tetrapoint, "eve@company.com")
    print(f"Eve: {user5_tetrapoint}")
    
    print("\n🔍 Querying with Tetralemma logic:")
    
    # Query all users
    print("\nAll users:")
    print(db.tetralemma_query("all")[['name', 'is_active_tetrapoint']])
    
    # Query active users (affirmation expressed)
    print("\nUsers with EXPRESSED affirmation (definitely active):")
    active_users = db.tetralemma_query("active")
    print(active_users[['name', 'is_active_tetrapoint']])
    
    # Query inactive users (negation expressed)
    print("\nUsers with EXPRESSED negation (definitely inactive):")
    inactive_users = db.tetralemma_query("inactive")
    print(inactive_users[['name', 'is_active_tetrapoint']])
    
    # Query contradictory users (both expressed)
    print("\nUsers with EXPRESSED both (contradictory information):")
    contradictory_users = db.tetralemma_query("contradictory")
    print(contradictory_users[['name', 'is_active_tetrapoint']])
    
    # Query unknown users (neither expressed)
    print("\nUsers with EXPRESSED neither (unknown status):")
    unknown_users = db.tetralemma_query("unknown")
    print(unknown_users[['name', 'is_active_tetrapoint']])
    
    # Show contradictions summary
    print("\n📋 Contradictions Summary:")
    contradictions = db.get_contradictions_summary()
    print(contradictions)
    
    print("\n💡 Key Benefits of Tetralemma Database:")
    print("1. Handles contradictory data without failing")
    print("2. Tracks uncertainty and missing information explicitly")
    print("3. Provides rich querying capabilities for all four logical states")
    print("4. Maintains data provenance and contradiction history")
    print("5. Supports gradual resolution of contradictions")
    
    return db

def tetralemma_aggregation_example():
    """Demonstrate aggregation operations with Tetralemma logic"""
    
    print("\n📈 Tetralemma Aggregation Example:")
    print("=" * 40)
    
    db = TetralemmaDatabase()
    
    # Insert sample data for aggregation
    users_data = [
        ("User1", Tetrapoint(TetralemmaValue.EXPRESSED, TetralemmaValue.SUPPRESSED, TetralemmaValue.SUPPRESSED, TetralemmaValue.SUPPRESSED)),
        ("User2", Tetrapoint(TetralemmaValue.SUPPRESSED, TetralemmaValue.EXPRESSED, TetralemmaValue.SUPPRESSED, TetralemmaValue.SUPPRESSED)),
        ("User3", Tetrapoint(TetralemmaValue.EXPRESSED, TetralemmaValue.EXPRESSED, TetralemmaValue.EXPRESSED, TetralemmaValue.SUPPRESSED)),
        ("User4", Tetrapoint(TetralemmaValue.SUPPRESSED, TetralemmaValue.SUPPRESSED, TetralemmaValue.SUPPRESSED, TetralemmaValue.EXPRESSED)),
        ("User5", Tetrapoint(TetralemmaValue.EXPRESSED, TetralemmaValue.SUPPRESSED, TetralemmaValue.SUPPRESSED, TetralemmaValue.SUPPRESSED)),
    ]
    
    for name, tetrapoint in users_data:
        db.insert_user(name, tetrapoint)
    
    # Aggregate statistics
    total_users = len(users_data)
    active_users = len(db.tetralemma_query("active"))
    inactive_users = len(db.tetralemma_query("inactive"))
    contradictory_users = len(db.tetralemma_query("contradictory"))
    unknown_users = len(db.tetralemma_query("unknown"))
    
    print(f"Total Users: {total_users}")
    print(f"Definitely Active: {active_users}")
    print(f"Definitely Inactive: {inactive_users}")
    print(f"Contradictory Status: {contradictory_users}")
    print(f"Unknown Status: {unknown_users}")
    
    print(f"\nActive Rate (excluding contradictions/unknown): {active_users/(active_users+inactive_users)*100:.1f}%")
    print(f"Contradiction Rate: {contradictory_users/total_users*100:.1f}%")
    print(f"Unknown Rate: {unknown_users/total_users*100:.1f}%")

if __name__ == "__main__":
    # Run the demonstration
    db = demonstrate_tetralemma_database()
    tetralemma_aggregation_example()
    
    print("\n✅ Tetralemma Database demonstration completed!")
    print("This system shows how four-valued logic can handle real-world data inconsistencies.") 