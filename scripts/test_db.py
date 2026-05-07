import os
import sys

# Add backend to path so we can import app
sys.path.append(os.path.join(os.getcwd(), "backend"))

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.core.database import engine, init_db
from app.models.db_models import User, RunLog, EventLog
from app.api.routes.auth import hash_password

def test_db():
    print("🚀 Initializing database (Pure SQLAlchemy)...")
    init_db()
    
    with Session(engine) as session:
        # 1. Create a test user
        print("👤 Creating test user...")
        # Check if user already exists
        test_user = session.execute(select(User).where(User.username == "test_user")).scalar_one_or_none()
        
        if not test_user:
            test_user = User(
                username="test_user",
                hashed_password=hash_password("password123")
            )
            session.add(test_user)
            session.commit()
            session.refresh(test_user)
            print(f"✅ User created: {test_user.username}")
        else:
            print(f"ℹ️ User already exists: {test_user.username}")

        # 2. Create a test run
        print("🏃 Creating test run...")
        test_run = RunLog(
            issue_url="https://github.com/owner/repo/issues/1",
            status="SUCCESS",
            user_id=test_user.id
        )
        session.add(test_run)
        session.commit()
        session.refresh(test_run)
        print(f"✅ Run created with ID: {test_run.run_id}")

        # 3. Create a test event
        print("📝 Creating test event...")
        test_event = EventLog(
            run_id=test_run.id,
            stage="CLONING",
            message="Cloning repository..."
        )
        session.add(test_event)
        session.commit()
        print("✅ Event created and linked to run")

        # 4. Verify retrieval
        print("🔍 Verifying data retrieval...")
        runs = session.execute(select(RunLog).where(RunLog.user_id == test_user.id)).scalars().all()
        print(f"✅ Found {len(runs)} runs for user {test_user.username}")
        
        for r in runs:
            # We need to refresh to access events if using lazy loading
            print(f"   - Run {r.run_id}: {r.status} ({len(r.events)} events)")

if __name__ == "__main__":
    test_db()
