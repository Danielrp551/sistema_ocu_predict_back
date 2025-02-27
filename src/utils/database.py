from ..extensions import db

def commit_changes():
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise e
