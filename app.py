import streamlit as st
import pickle
import re
import nltk
import fitz  # PyMuPDF

nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

# Resume text cleaning function
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Category mapping
category_mapping = {
    6: 'Data Science', 
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing',
}

# Streamlit app
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    if upload_file is not None:
        if upload_file.type == "application/pdf":
            with fitz.open(stream=upload_file.read(), filetype="pdf") as doc:
                resume_text = ""
                for page in doc:
                    resume_text += page.get_text()
        else:
            try:
                resume_text = upload_file.read().decode('utf-8')
            except UnicodeDecodeError:
                resume_text = upload_file.read().decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.success(f"Predicted Category ID: {prediction_id}")
        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()