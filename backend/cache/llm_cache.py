import streamlit as st
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache


@st.cache_resource
def configure_llm_cache():
    cache = InMemoryCache()
    set_llm_cache(cache)
    return cache
