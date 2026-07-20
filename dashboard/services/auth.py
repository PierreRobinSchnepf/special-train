"""Verrou admin pour les opérations d'écriture du dashboard en déploiement.

En local (pas de secrets S3), tout est débloqué : rien ne change pour le
développement. En cloud, les boutons qui *écrivent* ou déclenchent des travaux
lourds (actualisation des données, prévision réelle live) sont masqués tant que
le mot de passe admin (secret `admin_password`) n'a pas été saisi.
"""
from __future__ import annotations

import streamlit as st

from src.storage import admin_password, admin_unlocked, is_cloud


def admin_gate() -> bool:
    """Affiche (en cloud) un champ mot de passe dans la sidebar et renvoie True
    si l'utilisateur est débloqué. En local, renvoie True sans rien afficher."""
    if not is_cloud():
        return True

    if admin_unlocked():
        st.sidebar.success("🔓 Mode admin")
        return True

    with st.sidebar.expander("🔒 Accès admin"):
        pwd = st.text_input("Mot de passe", type="password", key="_admin_pwd_input")
        if st.button("Déverrouiller", key="_admin_unlock_btn"):
            expected = admin_password()
            if expected and pwd == expected:
                st.session_state["_admin_ok"] = True
                st.rerun()
            else:
                st.error("Mot de passe incorrect.")
    return False
