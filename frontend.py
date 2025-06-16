import streamlit as st
from document import process_pdf

def main():
    st.title("File Upload and Process Example")
    language_options = {
        'eng': 'English',
        'fra': 'French',
        'jpn': 'Japanese',
        'rus': 'Russian',
        'spa': 'Spanish',
        'heb': 'Hebrew'
    }

    selected_lang_code = st.selectbox(
        "Select OCR Language",
        options=list(language_options.keys()),
        format_func=lambda x: language_options[x]
    )

    uploaded_file = st.file_uploader("Upload a file", type=None)  # accept all types, restrict if needed

    if uploaded_file is not None:
        if st.button("Submit"):
            with st.spinner("Processing PDF... Please wait."):
                message = process_pdf(uploaded_file, selected_lang_code)
            print(message)
            translated_pdf_url = message.get("translated_pdf_url", "")

# Optional: strip trailing } if present
            translated_pdf_url = translated_pdf_url.rstrip('}')

# Display clickable link in Streamlit
            st.markdown(f"[Download Translated PDF]({translated_pdf_url})", unsafe_allow_html=True)

            # st.markdown(message)

if __name__ == "__main__":
    main()
