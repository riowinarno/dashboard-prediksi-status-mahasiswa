import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# =========================
# Load model dan encoder
# =========================
model = joblib.load("model_logreg_students.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Prediksi Status Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

# =========================
# Dokumentasi field
# =========================
field_descriptions = {
    "Marital_status": "Marital status: 1 = single, 2 = married, 3 = widower, 4 = divorced, 5 = facto union, 6 = legally separated.",

    "Application_mode": "Application mode based on the admission route code. Example codes include: 1 = 1st phase - general contingent, 2 = Ordinance No. 612/93, 5 = 1st phase - special contingent (Azores Island), 7 = holders of other higher courses, 10 = Ordinance No. 854-B/99, 15 = international student (bachelor), 16 = 1st phase - special contingent (Madeira Island), 17 = 2nd phase - general contingent, 18 = 3rd phase - general contingent, 26 = Ordinance No. 533-A/99, item b2, 27 = Ordinance No. 533-A/99, item b3, 39 = over 23 years old, 42 = transfer, 43 = change of course, 44 = technological specialization diploma holders, 51 = change of institution/course, 53 = short cycle diploma holders, 57 = change of institution/course (international).",

    "Application_order": "Application order between 0 and 9, where 0 indicates first choice and 9 indicates last choice.",

    "Course": "Course code of the student. Example codes include: 33 = Biofuel Production Technologies, 171 = Animation and Multimedia Design, 8014 = Social Service (evening attendance), 9003 = Agronomy, 9070 = Communication Design, 9085 = Veterinary Nursing, 9119 = Informatics Engineering, 9130 = Equinculture, 9147 = Management, 9238 = Social Service, 9254 = Tourism, 9500 = Nursing, 9556 = Oral Hygiene, 9670 = Advertising and Marketing Management, 9773 = Journalism and Communication, 9853 = Basic Education, 9991 = Management (evening attendance).",

    "Daytime_evening_attendance": "Attendance schedule: 1 = daytime, 0 = evening.",

    "Previous_qualification": "Previous qualification code. Example: 1 = secondary education, 2 = higher education - bachelor's degree, 3 = higher education - degree, 4 = higher education - master's, 5 = higher education - doctorate, 9 = 12th year of schooling - not completed, 10 = 11th year of schooling - not completed, 12 = other - 11th year of schooling, 14 = 10th year of schooling, 15 = 10th year of schooling - not completed, 19 = basic education 3rd cycle (9th/10th/11th year) or equivalent, 38 = basic education 2nd cycle (6th/7th/8th year) or equivalent, 39 = technological specialization course, 40 = higher education - degree (1st cycle), 42 = professional higher technical course, 43 = higher education - master (2nd cycle).",

    "Previous_qualification_grade": "Grade of previous qualification, ranging between 0 and 200.",

    "Nacionality": "Nationality code. Example: 1 = Portuguese, 2 = German, 6 = Spanish, 11 = Italian, 13 = Dutch, 14 = English, 17 = Lithuanian, 21 = Angolan, 22 = Cape Verdean, 24 = Guinean, 25 = Mozambican, 26 = Santomean, 32 = Turkish, 41 = Brazilian, 62 = Romanian, 100 = Moldovan, 101 = Mexican, 103 = Ukrainian, and others according to the dataset documentation.",

    "Mothers_qualification": "Mother's qualification level based on coded categories in the dataset documentation.",

    "Fathers_qualification": "Father's qualification level based on coded categories in the dataset documentation.",

    "Mothers_occupation": "Mother's occupation based on coded categories in the dataset documentation.",

    "Fathers_occupation": "Father's occupation based on coded categories in the dataset documentation.",

    "Admission_grade": "Admission grade, ranging between 0 and 200.",

    "Displaced": "Displaced student indicator: 1 = yes, 0 = no.",

    "Educational_special_needs": "Educational special needs indicator: 1 = yes, 0 = no.",

    "Debtor": "Debtor status indicator: 1 = yes, 0 = no.",

    "Tuition_fees_up_to_date": "Tuition fees payment status: 1 = yes (up to date), 0 = no.",

    "Gender": "Gender: 1 = male, 0 = female.",

    "Scholarship_holder": "Scholarship holder indicator: 1 = yes, 0 = no.",

    "Age_at_enrollment": "Age of the student at the time of enrollment.",

    "International": "International student indicator: 1 = yes, 0 = no.",

    "Curricular_units_1st_sem_credited": "Number of curricular units credited in the 1st semester.",

    "Curricular_units_1st_sem_enrolled": "Number of curricular units enrolled in the 1st semester.",

    "Curricular_units_1st_sem_evaluations": "Number of evaluations to curricular units in the 1st semester.",

    "Curricular_units_1st_sem_approved": "Number of curricular units approved in the 1st semester.",

    "Curricular_units_1st_sem_grade": "Grade average in the 1st semester, ranging between 0 and 20.",

    "Curricular_units_1st_sem_without_evaluations": "Number of curricular units without evaluations in the 1st semester.",

    "Curricular_units_2nd_sem_credited": "Number of curricular units credited in the 2nd semester.",

    "Curricular_units_2nd_sem_enrolled": "Number of curricular units enrolled in the 2nd semester.",

    "Curricular_units_2nd_sem_evaluations": "Number of evaluations to curricular units in the 2nd semester.",

    "Curricular_units_2nd_sem_approved": "Number of curricular units approved in the 2nd semester.",

    "Curricular_units_2nd_sem_grade": "Grade average in the 2nd semester, ranging between 0 and 20.",

    "Curricular_units_2nd_sem_without_evaluations": "Number of curricular units without evaluations in the 2nd semester.",

    "Unemployment_rate": "Unemployment rate (%).",

    "Inflation_rate": "Inflation rate (%).",

    "GDP": "Gross Domestic Product (GDP)."
}

field_labels = {
    "Marital_status": "Marital Status",
    "Application_mode": "Application Mode",
    "Application_order": "Application Order",
    "Course": "Course",
    "Daytime_evening_attendance": "Daytime / Evening Attendance",
    "Previous_qualification": "Previous Qualification",
    "Previous_qualification_grade": "Previous Qualification Grade",
    "Nacionality": "Nationality",
    "Mothers_qualification": "Mother's Qualification",
    "Fathers_qualification": "Father's Qualification",
    "Mothers_occupation": "Mother's Occupation",
    "Fathers_occupation": "Father's Occupation",
    "Admission_grade": "Admission Grade",
    "Displaced": "Displaced",
    "Educational_special_needs": "Educational Special Needs",
    "Debtor": "Debtor",
    "Tuition_fees_up_to_date": "Tuition Fees Up To Date",
    "Gender": "Gender",
    "Scholarship_holder": "Scholarship Holder",
    "Age_at_enrollment": "Age at Enrollment",
    "International": "International Student",
    "Curricular_units_1st_sem_credited": "1st Semester Credited Units",
    "Curricular_units_1st_sem_enrolled": "1st Semester Enrolled Units",
    "Curricular_units_1st_sem_evaluations": "1st Semester Evaluations",
    "Curricular_units_1st_sem_approved": "1st Semester Approved Units",
    "Curricular_units_1st_sem_grade": "1st Semester Grade",
    "Curricular_units_1st_sem_without_evaluations": "1st Semester Without Evaluations",
    "Curricular_units_2nd_sem_credited": "2nd Semester Credited Units",
    "Curricular_units_2nd_sem_enrolled": "2nd Semester Enrolled Units",
    "Curricular_units_2nd_sem_evaluations": "2nd Semester Evaluations",
    "Curricular_units_2nd_sem_approved": "2nd Semester Approved Units",
    "Curricular_units_2nd_sem_grade": "2nd Semester Grade",
    "Curricular_units_2nd_sem_without_evaluations": "2nd Semester Without Evaluations",
    "Unemployment_rate": "Unemployment Rate",
    "Inflation_rate": "Inflation Rate",
    "GDP": "GDP"
}

# =========================
# Mapping label
# =========================
marital_status_map = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto Union",
    6: "Legally Separated"
}

binary_yes_no_map = {
    0: "No",
    1: "Yes"
}

gender_map = {
    0: "Female",
    1: "Male"
}

attendance_map = {
    0: "Evening",
    1: "Daytime"
}

# =========================
# Header
# =========================
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("""
Aplikasi ini digunakan untuk memprediksi status mahasiswa menjadi:
**Dropout**, **Enrolled**, atau **Graduate**  
berdasarkan data demografis, administratif, dan performa akademik.
""")

st.info("Silakan isi data mahasiswa pada form di bawah ini, lalu klik tombol **Prediksi Status**.")

# =========================
# Dokumentasi field
# =========================
with st.expander("📘 Lihat dokumentasi field input"):
    st.markdown("Berikut penjelasan singkat untuk setiap field yang digunakan dalam model:")
    for field, desc in field_descriptions.items():
        st.markdown(f"**{field_labels[field]}**: {desc}")

# =========================
# Sidebar info
# =========================
st.sidebar.header("ℹ️ Informasi Aplikasi")
st.sidebar.write(
    "Prototype ini menggunakan model **Logistic Regression** untuk memprediksi status mahasiswa."
)

selected_field = st.sidebar.selectbox(
    "Pilih field untuk melihat dokumentasi",
    list(field_descriptions.keys()),
    format_func=lambda x: field_labels[x]
)

st.sidebar.markdown(f"**{field_labels[selected_field]}**")
st.sidebar.write(field_descriptions[selected_field])

# =========================
# Form input
# =========================
with st.form("prediction_form"):
    st.subheader("Input Data Mahasiswa")

    col1, col2, col3 = st.columns(3)

    with col1:
        marital_status = st.selectbox(
            "Marital Status",
            options=list(marital_status_map.keys()),
            format_func=lambda x: marital_status_map[x],
            help=field_descriptions["Marital_status"]
        )

        application_mode = st.number_input(
            "Application Mode",
            min_value=1,
            max_value=60,
            value=1,
            help=field_descriptions["Application_mode"]
        )

        application_order = st.number_input(
            "Application Order",
            min_value=0,
            max_value=9,
            value=1,
            help=field_descriptions["Application_order"]
        )

        course = st.number_input(
            "Course",
            min_value=1,
            max_value=1000,
            value=33,
            help=field_descriptions["Course"]
        )

        daytime_evening_attendance = st.selectbox(
            "Daytime / Evening Attendance",
            options=list(attendance_map.keys()),
            format_func=lambda x: attendance_map[x],
            help=field_descriptions["Daytime_evening_attendance"]
        )

        previous_qualification = st.number_input(
            "Previous Qualification",
            min_value=1,
            max_value=50,
            value=1,
            help=field_descriptions["Previous_qualification"]
        )

        previous_qualification_grade = st.number_input(
            "Previous Qualification Grade",
            min_value=0.0,
            max_value=200.0,
            value=130.0,
            help=field_descriptions["Previous_qualification_grade"]
        )

        nationality = st.number_input(
            "Nationality",
            min_value=1,
            max_value=200,
            value=1,
            help=field_descriptions["Nacionality"]
        )

        mothers_qualification = st.number_input(
            "Mother's Qualification",
            min_value=1,
            max_value=50,
            value=1,
            help=field_descriptions["Mothers_qualification"]
        )

        fathers_qualification = st.number_input(
            "Father's Qualification",
            min_value=1,
            max_value=50,
            value=1,
            help=field_descriptions["Fathers_qualification"]
        )

        mothers_occupation = st.number_input(
            "Mother's Occupation",
            min_value=0,
            max_value=200,
            value=0,
            help=field_descriptions["Mothers_occupation"]
        )

        fathers_occupation = st.number_input(
            "Father's Occupation",
            min_value=0,
            max_value=200,
            value=0,
            help=field_descriptions["Fathers_occupation"]
        )

    with col2:
        admission_grade = st.number_input(
            "Admission Grade",
            min_value=0.0,
            max_value=200.0,
            value=120.0,
            help=field_descriptions["Admission_grade"]
        )

        displaced = st.selectbox(
            "Displaced",
            options=list(binary_yes_no_map.keys()),
            format_func=lambda x: binary_yes_no_map[x],
            help=field_descriptions["Displaced"]
        )

        educational_special_needs = st.selectbox(
            "Educational Special Needs",
            options=list(binary_yes_no_map.keys()),
            format_func=lambda x: binary_yes_no_map[x],
            help=field_descriptions["Educational_special_needs"]
        )

        debtor = st.selectbox(
            "Debtor",
            options=list(binary_yes_no_map.keys()),
            format_func=lambda x: binary_yes_no_map[x],
            help=field_descriptions["Debtor"]
        )

        tuition_fees_up_to_date = st.selectbox(
            "Tuition Fees Up To Date",
            options=list(binary_yes_no_map.keys()),
            format_func=lambda x: binary_yes_no_map[x],
            help=field_descriptions["Tuition_fees_up_to_date"]
        )

        gender = st.selectbox(
            "Gender",
            options=list(gender_map.keys()),
            format_func=lambda x: gender_map[x],
            help=field_descriptions["Gender"]
        )

        scholarship_holder = st.selectbox(
            "Scholarship Holder",
            options=list(binary_yes_no_map.keys()),
            format_func=lambda x: binary_yes_no_map[x],
            help=field_descriptions["Scholarship_holder"]
        )

        age_at_enrollment = st.number_input(
            "Age at Enrollment",
            min_value=15,
            max_value=80,
            value=20,
            help=field_descriptions["Age_at_enrollment"]
        )

        international = st.selectbox(
            "International Student",
            options=list(binary_yes_no_map.keys()),
            format_func=lambda x: binary_yes_no_map[x],
            help=field_descriptions["International"]
        )

        unemployment_rate = st.number_input(
            "Unemployment Rate",
            min_value=-10.0,
            max_value=100.0,
            value=10.8,
            help=field_descriptions["Unemployment_rate"]
        )

        inflation_rate = st.number_input(
            "Inflation Rate",
            min_value=-10.0,
            max_value=100.0,
            value=1.4,
            help=field_descriptions["Inflation_rate"]
        )

        gdp = st.number_input(
            "GDP",
            min_value=-10.0,
            max_value=100.0,
            value=1.74,
            help=field_descriptions["GDP"]
        )

    with col3:
        curricular_units_1st_sem_credited = st.number_input(
            "1st Sem Credited Units",
            min_value=0,
            max_value=30,
            value=0,
            help=field_descriptions["Curricular_units_1st_sem_credited"]
        )

        curricular_units_1st_sem_enrolled = st.number_input(
            "1st Sem Enrolled Units",
            min_value=0,
            max_value=30,
            value=6,
            help=field_descriptions["Curricular_units_1st_sem_enrolled"]
        )

        curricular_units_1st_sem_evaluations = st.number_input(
            "1st Sem Evaluations",
            min_value=0,
            max_value=50,
            value=6,
            help=field_descriptions["Curricular_units_1st_sem_evaluations"]
        )

        curricular_units_1st_sem_approved = st.number_input(
            "1st Sem Approved Units",
            min_value=0,
            max_value=30,
            value=5,
            help=field_descriptions["Curricular_units_1st_sem_approved"]
        )

        curricular_units_1st_sem_grade = st.number_input(
            "1st Sem Grade",
            min_value=0.0,
            max_value=20.0,
            value=11.0,
            help=field_descriptions["Curricular_units_1st_sem_grade"]
        )

        curricular_units_1st_sem_without_evaluations = st.number_input(
            "1st Sem Without Evaluations",
            min_value=0,
            max_value=30,
            value=0,
            help=field_descriptions["Curricular_units_1st_sem_without_evaluations"]
        )

        curricular_units_2nd_sem_credited = st.number_input(
            "2nd Sem Credited Units",
            min_value=0,
            max_value=30,
            value=0,
            help=field_descriptions["Curricular_units_2nd_sem_credited"]
        )

        curricular_units_2nd_sem_enrolled = st.number_input(
            "2nd Sem Enrolled Units",
            min_value=0,
            max_value=30,
            value=6,
            help=field_descriptions["Curricular_units_2nd_sem_enrolled"]
        )

        curricular_units_2nd_sem_evaluations = st.number_input(
            "2nd Sem Evaluations",
            min_value=0,
            max_value=50,
            value=6,
            help=field_descriptions["Curricular_units_2nd_sem_evaluations"]
        )

        curricular_units_2nd_sem_approved = st.number_input(
            "2nd Sem Approved Units",
            min_value=0,
            max_value=30,
            value=5,
            help=field_descriptions["Curricular_units_2nd_sem_approved"]
        )

        curricular_units_2nd_sem_grade = st.number_input(
            "2nd Sem Grade",
            min_value=0.0,
            max_value=20.0,
            value=11.0,
            help=field_descriptions["Curricular_units_2nd_sem_grade"]
        )

        curricular_units_2nd_sem_without_evaluations = st.number_input(
            "2nd Sem Without Evaluations",
            min_value=0,
            max_value=30,
            value=0,
            help=field_descriptions["Curricular_units_2nd_sem_without_evaluations"]
        )

    submitted = st.form_submit_button("🔍 Prediksi Status")

# =========================
# Prediksi
# =========================
if submitted:
    input_data = pd.DataFrame([{
        'Marital_status': marital_status,
        'Application_mode': application_mode,
        'Application_order': application_order,
        'Course': course,
        'Daytime_evening_attendance': daytime_evening_attendance,
        'Previous_qualification': previous_qualification,
        'Previous_qualification_grade': previous_qualification_grade,
        'Nacionality': nationality,
        'Mothers_qualification': mothers_qualification,
        'Fathers_qualification': fathers_qualification,
        'Mothers_occupation': mothers_occupation,
        'Fathers_occupation': fathers_occupation,
        'Admission_grade': admission_grade,
        'Displaced': displaced,
        'Educational_special_needs': educational_special_needs,
        'Debtor': debtor,
        'Tuition_fees_up_to_date': tuition_fees_up_to_date,
        'Gender': gender,
        'Scholarship_holder': scholarship_holder,
        'Age_at_enrollment': age_at_enrollment,
        'International': international,
        'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
        'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
        'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
        'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
        'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
        'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_sem_without_evaluations,
        'Curricular_units_2nd_sem_credited': curricular_units_2nd_sem_credited,
        'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
        'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
        'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
        'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
        'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations,
        'Unemployment_rate': unemployment_rate,
        'Inflation_rate': inflation_rate,
        'GDP': gdp
    }])

    prediction = model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    st.subheader("📌 Hasil Prediksi")

    if prediction_label == "Dropout":
        st.error(f"Prediksi status mahasiswa: **{prediction_label}**")
        st.markdown("Mahasiswa terindikasi memiliki risiko tinggi untuk dropout dan memerlukan perhatian lebih lanjut.")
    elif prediction_label == "Enrolled":
        st.warning(f"Prediksi status mahasiswa: **{prediction_label}**")
        st.markdown("Mahasiswa diprediksi masih berada dalam status aktif/enrolled.")
    else:
        st.success(f"Prediksi status mahasiswa: **{prediction_label}**")
        st.markdown("Mahasiswa diprediksi memiliki peluang yang baik untuk graduate.")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        prob_df = pd.DataFrame({
            "Status": label_encoder.classes_,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        st.subheader("📊 Probabilitas Prediksi")
        st.dataframe(prob_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(prob_df["Status"], prob_df["Probability"], color="skyblue")
        plt.xticks(rotation=0)
        plt.ylabel("Probability")
        plt.xlabel("Status")
        plt.title("Probabilitas Prediksi")
        st.pyplot(fig)

    st.subheader("🧾 Ringkasan Data Input")
    st.dataframe(input_data, use_container_width=True)