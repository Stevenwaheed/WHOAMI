import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import seaborn as sns
from Bio import SeqIO
from io import StringIO

# Predict Data.

LR_file = 'logistic_regression_model.h5'
RF_file = 'Random_Forest_model.h5'
cv_file = 'count_vectorized.sav'


def getKmers(sequence, size=6):
    return [sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)]


menu = ["ML Models", "Similarity"]
choice_menu = st.sidebar.selectbox("Select Option", menu)

if choice_menu == 'ML Models':
    st.title('Classification Problems')
    file = st.file_uploader("Upload file", type=["fas", "fasta", 'txt'])

    if file is not None:
        options = st.radio("Machine Learning Models", options=["Classify Sequence", "Train New Model"])
        if options == "Classify Sequence":
            model = st.file_uploader("Upload model", type=["h5", "sav"])
            cv_file = st.file_uploader("Upload count vectorizer", type=["h5", "sav"])

            try:

                if model is not None and cv_file is not None:
                    loaded_LR_model = joblib.load(model)
                    loaded_cv_model = joblib.load(cv_file)

                    stringio = StringIO(file.getvalue().decode("utf-8"))
                    loaded_file = SeqIO.parse(stringio, 'fasta')

                    sequence_list = []
                    for sequence in loaded_file:
                        sequence_list.append(str(sequence.seq))

                    fasta_df = pd.DataFrame(columns=['Sequences'])
                    fasta_df['Sequences'] = sequence_list

                    fasta_df['words'] = fasta_df.apply(lambda x: getKmers(x['Sequences']), axis=1)
                    fasta_df = fasta_df.drop('Sequences', axis=1)

                    fasta_texts = list(fasta_df['words'])
                    for item in range(len(fasta_texts)):
                        fasta_texts[item] = ' '.join(fasta_texts[item])

                    cv = CountVectorizer()

                    X_fasta = loaded_cv_model.transform(fasta_texts)

                    fasta_file_preds = loaded_LR_model.predict(X_fasta)
                    lst_preds = []
                    for pred in fasta_file_preds:
                        if pred == 0:
                            pred = 'G protein coupled'
                        elif pred == 1:
                            pred = 'Tyrosine kinase'
                        elif pred == 2:
                            pred = 'Tyrosine phosphatase'
                        elif pred == 3:
                            pred = 'Synthetase'
                        elif pred == 4:
                            pred = 'Synthase'
                        elif pred == 5:
                            pred = 'Ion channel'
                        elif pred == 6:
                            pred = 'Transcription factor'
                        lst_preds.append(pred)

                    df = pd.DataFrame(columns=['Sequence', 'Class'])
                    df['Sequence'] = sequence_list
                    df['Class'] = lst_preds

                    st.dataframe(df.head(20), 1000, 700)

            except:
                st.warning('Invalid File...')

        elif options == "Train New Model":
            try:

                st.write('Reading the data...')

                data = pd.read_table(file)
                st.table(data.head())
                st.write('Preprocessing...')

                data['words'] = data.apply(lambda x: getKmers(x['sequence']), axis=1)
                data = data.drop('sequence', axis=1)

                data_texts = list(data['words'])
                for item in range(len(data_texts)):
                    data_texts[item] = ' '.join(data_texts[item])
                y_data = data.iloc[:, 0].values

                data.loc[data['class'] == 0, 'class'] = 'G protein coupled'
                data.loc[data['class'] == 1, 'class'] = 'Tyrosine kinase'
                data.loc[data['class'] == 2, 'class'] = 'Tyrosine phosphatase'
                data.loc[data['class'] == 3, 'class'] = 'Synthetase'
                data.loc[data['class'] == 4, 'class'] = 'Synthase'
                data.loc[data['class'] == 5, 'class'] = 'Ion channel'
                data.loc[data['class'] == 6, 'class'] = 'Transcription factor'

                cv = CountVectorizer()
                X = cv.fit_transform(data_texts)

                X_train, X_test, y_train, y_test = train_test_split(X,
                                                                    y_data,
                                                                    test_size=0.20,
                                                                    random_state=42)

                models = st.radio("Models", options=["Logistic Regression", "Random Forest"])
                if models == 'Logistic Regression':

                    choices_penalty = st.radio('Penalty Choices', options=['l1', 'l2', 'elasticnet'])

                    choices_solver = st.radio('Solver Choices', options=['lbfgs', 'liblinear', 'sag', 'newton-cg'
                                                                        , 'newton-cholesky', 'saga'])

                    choices_max_iter = int(st.text_input('Enter the maximum iterations (Integer number)'))

                    if choices_max_iter <= 0:
                        choices_max_iter = 10

                    LR = LogisticRegression(max_iter=choices_max_iter, penalty=choices_penalty, solver=choices_solver)
                    LR.fit(X_train, y_train)

                    preds = LR.predict(X_test)

                    st.write('Preparing the Model...')

                    st.subheader('Model performance')
                    performance = pd.DataFrame(columns=['accuracy_score', 'f1_score', 'precision_score', 'recall_score'])
                    performance['accuracy_score'] = [accuracy_score(y_test, preds)]
                    performance['f1_score'] = [f1_score(y_test, preds, average='weighted')]
                    performance['precision_score'] = [precision_score(y_test, preds, average='weighted')]
                    performance['recall_score'] = [recall_score(y_test, preds, average='weighted')]

                    st.dataframe(performance, 1000, 50)

                    joblib.dump(LR, LR_file)
                    joblib.dump(cv, cv_file)

                elif models == 'Random Forest':
                    choices_criterion = st.radio('Criterion Choices', options=['gini', 'entropy', 'log_loss'])

                    choices_max_features = st.radio('Max Features Choices', options=['sqrt', 'log2'])

                    choices_n_estimators = int(st.text_input('Enter the number of estimators (Integer number)'))
                    choices_max_depth = int(st.text_input('Enter the maximum depth (Integer number)'))

                    if choices_n_estimators <= 0:
                        choices_n_estimators = 10

                    if choices_max_depth <= 0:
                        choices_max_depth = 10

                    RF = RandomForestClassifier(max_features=choices_max_features, max_depth=choices_max_depth, n_estimators=choices_n_estimators, criterion=choices_criterion, )
                    RF.fit(X_train, y_train)

                    preds = RF.predict(X_test)

                    st.write('Preparing the Model...')

                    st.subheader('Model performance')
                    performance = pd.DataFrame(columns=['accuracy_score', 'f1_score', 'precision_score', 'recall_score'])
                    performance['accuracy_score'] = [accuracy_score(y_test, preds)]
                    performance['f1_score'] = [f1_score(y_test, preds, average='weighted')]
                    performance['precision_score'] = [precision_score(y_test, preds, average='weighted')]
                    performance['recall_score'] = [recall_score(y_test, preds, average='weighted')]

                    st.dataframe(performance, 1000, 50)

                    joblib.dump(RF, RF_file)
                    joblib.dump(cv, cv_file)

            except:
                st.warning('Invalid File...')

# st.write('--------------------------------------------------')

elif choice_menu == 'Similarity':
    st.title('Similarity Problems')
    st.text('Compare your file sequences with Human, Dog and Chimpanzee genes')

    file_similarity = st.file_uploader("Upload fasta file", type=["fas", "fasta", 'txt'])

    if file_similarity is not None:
        human_data = pd.read_table('human_data.txt')
        chimp_data = pd.read_table('chimp_data.txt')
        dog_data = pd.read_table('dog_data.txt')

        human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
        human_data = human_data.drop('sequence', axis=1)

        chimp_data['words'] = chimp_data.apply(lambda x: getKmers(x['sequence']), axis=1)
        chimp_data = chimp_data.drop('sequence', axis=1)

        dog_data['words'] = dog_data.apply(lambda x: getKmers(x['sequence']), axis=1)
        dog_data = dog_data.drop('sequence', axis=1)

        human_texts = list(human_data['words'])
        for item in range(len(human_texts)):
            human_texts[item] = ' '.join(human_texts[item])

        chimp_texts = list(chimp_data['words'])
        for item in range(len(chimp_texts)):
            chimp_texts[item] = ' '.join(chimp_texts[item])

        dog_texts = list(dog_data['words'])
        for item in range(len(dog_texts)):
            dog_texts[item] = ' '.join(dog_texts[item])

        cv = CountVectorizer()
        human = cv.fit_transform(human_texts)
        chimp = cv.transform(chimp_texts)
        dog = cv.transform(dog_texts)

        stringio = StringIO(file_similarity.getvalue().decode("utf-8"))
        file = SeqIO.parse(stringio, 'fasta')

        sequences = []
        for line in file:
            sequences.append(str(line.seq))

        lst = []
        for i in sequences:
            lst.append(getKmers(i))

        df_seq = pd.DataFrame(columns=['seq'])
        df_seq['seq'] = lst

        seq = list(df_seq['seq'])
        for item in range(len(seq)):
            seq[item] = ' '.join(seq[item])

        cv_seq = cv.transform(seq)

        similarity_human_seq = cosine_similarity(cv_seq, human)
        similarity_dog_seq = cosine_similarity(cv_seq, chimp)
        similarity_chimp_seq = cosine_similarity(cv_seq, dog)

        similarity_df = pd.DataFrame(columns=['Sequence', 'Human', 'Dog', 'Chimpanzee'])

        st.subheader('Similarity with human')
        st.write(np.sort(similarity_human_seq))

        st.subheader('Similarity with dog')
        st.write(np.sort(similarity_dog_seq))

        st.subheader('Similarity with chimpanzee')
        st.write(np.sort(similarity_chimp_seq))

        similarity_df['Human'] = np.sort(similarity_human_seq.flatten())[-1:-300:-1]
        similarity_df['Dog'] = np.sort(similarity_dog_seq.flatten())[-1:-300:-1]
        similarity_df['Chimpanzee'] = np.sort(similarity_chimp_seq.flatten())[-1:-300:-1]

        # st.dataframe(similarity_df.head(20), 1000, 500)
        #
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.bar_chart(similarity_df)
