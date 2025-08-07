from datetime import datetime, timedelta

from flask import current_app, request, session, redirect, url_for

@current_app.before_request
def check_session():
    # 对于需要session的路由进行检查
    if request.endpoint in ['show_results']:
        if 'filenames' not in session or not session.get('analysis_complete', False):
            return redirect(url_for('index'))

@current_app.before_request
def check_session_expiry():
    if 'last_activity' in session:
        last_active = datetime.fromisoformat(session['last_activity'])
        if datetime.now() - last_active > timedelta(minutes=30):
            session.clear()
    session['last_activity'] = datetime.now().isoformat()



@current_app.before_request
def validate_session():
    if request.endpoint in ['show_results']:
        required_keys = ['filenames', 'file_metadata']
        if not all(k in session for k in required_keys):
            current_app.logger.warning(f"Missing session keys - required: {required_keys}")
            return redirect(url_for('index'))

@current_app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    return response

@current_app.before_request
def log_request():
    if request.path.startswith('/preview-report'):
        current_app.logger.info(f"Preview request: {request.path}")

