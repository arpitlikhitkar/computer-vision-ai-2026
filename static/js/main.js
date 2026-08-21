/* Main JavaScript Application Logic for Household AI Software */

document.addEventListener('DOMContentLoaded', () => {
    fetchEnrolledMembers();
    fetchRecentLogs();

    // Poll logs every 3 seconds
    setInterval(fetchRecentLogs, 3000);

    const enrollmentForm = document.getElementById('enrollmentForm');
    if (enrollmentForm) {
        enrollmentForm.addEventListener('submit', handleEnrollmentSubmit);
    }
});

// Fetch Enrolled Household Members
async function fetchEnrolledMembers() {
    try {
        const response = await fetch('/api/members');
        const data = await response.json();

        const membersList = document.getElementById('membersList');
        if (!membersList) return;

        if (data.members.length === 0) {
            membersList.innerHTML = `
                <div style="text-align: center; color: var(--text-secondary); padding: 20px;">
                    No enrolled members yet. Use the form below to enroll yourself!
                </div>
            `;
            return;
        }

        membersList.innerHTML = data.members.map(member => `
            <div class="member-item">
                <div class="member-info">
                    <h4>${member.display_name} <span style="font-size: 0.8rem; color: #818cf8;">(${member.display_id})</span></h4>
                    <p>Status: <span style="color: ${member.status === 'ACTIVE' ? '#10b981' : '#f43f5e'}; font-weight: 600;">${member.status}</span> | Embeddings: ${member.sample_count}</p>
                </div>
                <div class="member-actions">
                    <button class="btn-sm btn-secondary" onclick="toggleMemberStatus('${member.person_uuid}', '${member.status}')">
                        ${member.status === 'ACTIVE' ? 'Deactivate' : 'Activate'}
                    </button>
                    <button class="btn-sm btn-danger" onclick="deleteMember('${member.person_uuid}', '${member.display_name}')">
                        Delete
                    </button>
                </div>
            </div>
        `).join('');
    } catch (err) {
        console.error('Error fetching members:', err);
    }
}

// Fetch Recognition Audit Logs
async function fetchRecentLogs() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();

        const logsBody = document.getElementById('logsTableBody');
        if (!logsBody) return;

        if (data.logs.length === 0) {
            logsBody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-secondary);">No recognition logs recorded yet.</td></tr>`;
            return;
        }

        logsBody.innerHTML = data.logs.map(log => {
            const isKnown = log.recognition_result === 'KNOWN';
            const badgeClass = isKnown ? 'badge-known' : 'badge-unknown';
            const timeStr = log.timestamp ? log.timestamp.split('T')[1].split('.')[0] : 'N/A';

            return `
                <tr>
                    <td>${timeStr}</td>
                    <td>Track ${log.track_id}</td>
                    <td><span class="badge ${badgeClass}">${isKnown ? (log.person_name || 'Known') : 'UNKNOWN'}</span></td>
                    <td>${(log.similarity_score * 100).toFixed(1)}%</td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        console.error('Error fetching logs:', err);
    }
}

// Handle Interactive Enrollment Submit
async function handleEnrollmentSubmit(e) {
    e.preventDefault();
    const nameInput = document.getElementById('memberNameInput');
    const name = nameInput.value.trim();

    if (!name) {
        alert('Please enter member name.');
        return;
    }

    const btn = document.getElementById('startEnrollBtn');
    const statusMsg = document.getElementById('enrollStatusMsg');
    const progressBar = document.getElementById('progressBarFill');

    btn.disabled = true;
    btn.innerHTML = 'Enrolling... (Look at Camera)';
    statusMsg.innerText = 'Capturing 10 high-quality face samples... Please look directly at the camera.';
    progressBar.style.width = '10%';

    try {
        const response = await fetch('/api/enroll', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name })
        });

        const data = await response.json();

        if (data.success) {
            progressBar.style.width = '100%';
            statusMsg.innerHTML = `<span style="color: #10b981; font-weight: 600;">✓ Enrolled ${data.display_name} (${data.display_id}) with 10 face samples successfully!</span>`;
            nameInput.value = '';
            fetchEnrolledMembers();
        } else {
            statusMsg.innerHTML = `<span style="color: #f43f5e; font-weight: 600;">❌ Enrollment failed: ${data.message}</span>`;
            progressBar.style.width = '0%';
        }
    } catch (err) {
        statusMsg.innerHTML = `<span style="color: #f43f5e;">Error during enrollment: ${err.message}</span>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '✨ Enroll New Member';
    }
}

// Toggle Member Status (Active / Inactive)
async function toggleMemberStatus(uuid, currentStatus) {
    const newStatus = currentStatus === 'ACTIVE' ? 'INACTIVE' : 'ACTIVE';
    try {
        const response = await fetch('/api/members/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uuid: uuid, status: newStatus })
        });
        const data = await response.json();
        if (data.success) {
            fetchEnrolledMembers();
        }
    } catch (err) {
        console.error('Error toggling member status:', err);
    }
}

// Delete Member
async function deleteMember(uuid, name) {
    if (!confirm(`Are you sure you want to delete ${name} from enrolled members?`)) {
        return;
    }
    try {
        const response = await fetch('/api/members/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ uuid: uuid })
        });
        const data = await response.json();
        if (data.success) {
            fetchEnrolledMembers();
        }
    } catch (err) {
        console.error('Error deleting member:', err);
    }
}
