import streamlit_authenticator as stauth

passwords_to_hash = ['Agri@205@Vijay_$_*']
hashed_passwords = stauth.Hasher.hash_list(passwords_to_hash)
print(hashed_passwords[0])
