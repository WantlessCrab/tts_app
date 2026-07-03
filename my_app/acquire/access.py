# my_app/acquire/access.py
"""
Acquire service access policy.

Production acquisition is limited to validated direct PDF retrieval from the
supplied URL. Any source that requires interactive browser state or additional
user-mediated access must be supplied as a user-uploaded file.
"""