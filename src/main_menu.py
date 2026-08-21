"""
Household Face Recognition System — Main Interactive CLI Menu (Phase 5)

Options:
1. Run Real-Time Household Face Recognition
2. Enroll New Household Member
3. List All Enrolled Household Members
4. Deactivate / Activate Member
5. View Recent Recognition Audit Logs
6. Exit
"""

import sys
import os
from src.config import settings
from src.storage.database import initialize_database
from src.storage.person_repository import PersonRepository
from src.storage.embedding_repository import EmbeddingRepository
from src.storage.log_repository import LogRepository
from src.enrollment.enroll_person import run_interactive_enrollment
from src.phase5_face_recognition import run_phase5_face_recognition


def display_menu():
    print("\n==================================================")
    print(" HOUSEHOLD FACE RECOGNITION AI SYSTEM (PHASE 5)")
    print("==================================================")
    print(" 1. 🎥 Start Face Recognition Camera Stream")
    print(" 2. 👤 Enroll New Household Member")
    print(" 3. 📋 List All Enrolled Members")
    print(" 4. 🔒 Deactivate / Activate Member")
    print(" 5. 📜 View Recent Recognition Audit Logs")
    print(" 6. 🚪 Exit")
    print("==================================================")


def list_members(person_repo, embedding_repo):
    persons = person_repo.get_all_persons()
    print("\n==================================================")
    print(" ENROLLED HOUSEHOLD MEMBERS")
    print("==================================================")
    if not persons:
        print(" [INFO] No enrolled household members found in database.")
    else:
        for p in persons:
            vecs = embedding_repo.get_embeddings_for_person(p["person_uuid"])
            status_icon = "🟢 ACTIVE" if p["status"] == "ACTIVE" else "🔴 INACTIVE"
            print(f" {p['display_id']} | Name: {p['display_name']:<15} | Status: {status_icon} | Embeddings: {len(vecs)}")
    print("==================================================")


def toggle_status(person_repo):
    persons = person_repo.get_all_persons()
    if not persons:
        print("[INFO] No members available to modify.")
        return

    print("\nSelect Person to Toggle Status:")
    for idx, p in enumerate(persons, 1):
        print(f" {idx}. {p['display_id']} - {p['display_name']} ({p['status']})")

    choice = input("Enter choice number: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(persons):
        selected = persons[int(choice) - 1]
        new_status = "INACTIVE" if selected["status"] == "ACTIVE" else "ACTIVE"
        person_repo.update_person_status(selected["person_uuid"], new_status)
        print(f"[SUCCESS] Updated {selected['display_name']} status to {new_status}!")
    else:
        print("[ERROR] Invalid choice.")


def main():
    initialize_database()
    person_repo = PersonRepository()
    embedding_repo = EmbeddingRepository()
    log_repo = LogRepository()

    while True:
        display_menu()
        choice = input("Enter option number (1-6): ").strip()

        if choice == "1":
            run_phase5_face_recognition()
        elif choice == "2":
            run_interactive_enrollment()
        elif choice == "3":
            list_members(person_repo, embedding_repo)
        elif choice == "4":
            toggle_status(person_repo)
        elif choice == "5":
            logs = log_repo.get_recent_logs(10)
            print("\n==================================================")
            print(" RECENT RECOGNITION AUDIT LOGS")
            print("==================================================")
            if not logs:
                print(" [INFO] No recognition logs available.")
            else:
                for l in logs:
                    res_str = f"🟢 KNOWN" if l["recognition_result"] == "KNOWN" else "🔴 UNKNOWN"
                    print(f" Track {l['track_id']:<3} | Result: {res_str} | Similarity: {l['similarity_score']*100:.1f}% | Time: {l['timestamp'][:19]}")
            print("==================================================")
        elif choice == "6":
            print("[INFO] Exiting Household Face Recognition System. Goodbye!")
            sys.exit(0)
        else:
            print("[ERROR] Invalid option. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()
