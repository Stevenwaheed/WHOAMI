import streamlit as st
import neatbio.sequtils as utils
import io
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from Bio.Seq import Seq
from Bio import Seq
from Bio import SeqIO
from collections import Counter
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
from bs4 import BeautifulSoup
import requests
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from itertools import zip_longest

matplotlib.use("Agg")

LR_file = './logistic_regression_model.h5'
RF_file = './Random_Forest_model.h5'
cv_new_file = './count_vectorized.sav'


def getKmers(sequence, size=6):
    return [sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)]


def delta(x, y):
    return 0 if x == y else 1


def M(seq1, seq2, i, j, k):
    return sum(delta(x, y) for x, y in zip(seq1[i:i + k], seq2[j:j + k]))


def makeMatrix(seq1, seq2, k):
    n = len(seq1)
    m = len(seq2)
    return [[M(seq1, seq2, i, j, k) for j in range(m - k + 1)] for i in range(n - k + 1)]


def plotMatrix(M, t, seq1, seq2, nonblank=chr(0x25A0), blank=' '):
    print(' |' + seq2)
    print('-' * (2 + len(seq2)))
    for label, row in zip(seq1, M):
        line = ''.join(nonblank if s < t else blank for s in row)
        print(label + '|' + line)


def dotplot(seq1, seq2, k=1, t=1):
    M = makeMatrix(seq1, seq2, k)
    plotMatrix(M, t, seq1, seq2)  # experiment with character choice


# Convert to Fxn
def dotplotx(seq1, seq2):
    plt.imshow(np.array(makeMatrix(seq1, seq2, 1)))
    # on x-axis list all sequences of seq 2
    xt = plt.xticks(np.arange(len(list(seq2))), list(seq2))
    # on y-axis list all sequences of seq 1
    yt = plt.yticks(np.arange(len(list(seq1))), list(seq1))
    plt.show()


def gc_content(seq):
    result = float(str(seq).count('G') + str(seq).count('C')) / len(seq) * 100
    return result


def at_content(seq):
    result = float(str(seq).count('A') + str(seq).count('T')) / len(seq) * 100
    return result


def insertion_mutations(sequence):
    bases = 'ATCG'
    insert_bases = ''
    seq = ''
    range_mutation = random.randint(50, 100)

    for rand in range(range_mutation):
        random_number = random.randint(0, 1)
        if random_number == 0:
            for i in range(100):
                number = random.randint(0, 3)
                insert_bases += bases[number]

            index = random.randint(0, len(sequence) - 1)
            seq = sequence[:index] + insert_bases + sequence[index:]
        else:
            number = random.randint(0, 3)
            insert_bases += bases[number]
            index = random.randint(0, len(sequence) - 1)
            seq = sequence[:index] + insert_bases + sequence[index:]

    with st.expander("See Sequence"):
        st.write(seq)

    st.download_button("Download Insertion File", str(seq), file_name="insertion_mutations.txt")
    return seq


def deletion_mutations(sequence):
    seq = Seq.MutableSeq(sequence)

    range_mutation = random.randint(1000, 4000)
    for rand in range(range_mutation):
        number = random.randint(0, len(seq) - 1)
        seq.remove(seq[number])

    with st.expander("See Sequence"):
        st.write(seq)

    st.download_button("Download Deletion File", str(seq), file_name="deletion_mutations.txt")
    return seq


def substitution_mutations(sequence):
    seq = Seq.MutableSeq(sequence)
    bases = 'ATCG'
    number = random.randint(100, 300)
    for i in range(number):
        number_bases = random.randint(0, 3)
        number_seq = random.randint(0, len(seq))
        seq[number_seq] = bases[number_bases]

    with st.expander("See Sequence"):
        st.write(seq)

    st.download_button("Download Substitution File", str(seq), file_name="substitution_mutations.txt")
    return seq


def visualize_together(normal, mutation):
    sorted(normal)
    sorted(mutation)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("violinplot")
        plt.violinplot(list(normal.values()))
        plt.violinplot(list(mutation.values()))
        # plt.legend(['normal', 'mutation'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    with col2:
        st.header("plot")

        plt.plot(list(normal.values()))
        plt.plot(list(mutation.values()), linestyle='dashdot')
        plt.legend(['normal', 'mutation'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    with col3:
        st.header("boxplot")

        plt.boxplot(list(normal.values()))
        plt.boxplot(list(mutation.values()))
        # plt.legend(['normal', 'mutation'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


def count_dict(Translate):
    dict = {}
    for base in Translate.replace('*', ''):
        if base in dict:
            dict[base] += 1
        else:
            dict[base] = 0

    return dict


def main():
    # st.snow()
    image = Image.open('./logo/logo.png')

    st.image(image, use_column_width=True)

    # """A Simple Bioinformatics App"""
    st.title("WHOAMI - Sequencing App")

    menu = ["Intro", 'ML Models', 'Similarity', "DNA Sequence", "DotPlot", "Alignment", "DNA Mutation", "Blast", "About"]
    choice = st.sidebar.selectbox("Select Option", menu)

    if choice == "Intro":
        st.subheader("Python Project -Bioinformatics")

    elif choice == 'ML Models':
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

                        LR = LogisticRegression(max_iter=choices_max_iter, penalty=choices_penalty,
                                                solver=choices_solver)
                        LR.fit(X_train, y_train)

                        preds = LR.predict(X_test)

                        st.write('Preparing the Model...')

                        st.subheader('Model performance')
                        performance = pd.DataFrame(
                            columns=['accuracy_score', 'f1_score', 'precision_score', 'recall_score'])
                        performance['accuracy_score'] = [accuracy_score(y_test, preds)]
                        performance['f1_score'] = [f1_score(y_test, preds, average='weighted')]
                        performance['precision_score'] = [precision_score(y_test, preds, average='weighted')]
                        performance['recall_score'] = [recall_score(y_test, preds, average='weighted')]

                        st.dataframe(performance, 1000, 50)

                        joblib.dump(LR, LR_file)
                        joblib.dump(cv, cv_new_file)

                    elif models == 'Random Forest':
                        choices_criterion = st.radio('Criterion Choices', options=['gini', 'entropy', 'log_loss'])

                        choices_max_features = st.radio('Max Features Choices', options=['sqrt', 'log2'])

                        choices_n_estimators = int(st.text_input('Enter the number of estimators (Integer number)'))
                        choices_max_depth = int(st.text_input('Enter the maximum depth (Integer number)'))

                        if choices_n_estimators <= 0:
                            choices_n_estimators = 10

                        if choices_max_depth <= 0:
                            choices_max_depth = 10

                        RF = RandomForestClassifier(max_features=choices_max_features, max_depth=choices_max_depth,
                                                    n_estimators=choices_n_estimators, criterion=choices_criterion, )
                        RF.fit(X_train, y_train)

                        preds = RF.predict(X_test)

                        st.write('Preparing the Model...')

                        st.subheader('Model performance')
                        performance = pd.DataFrame(
                            columns=['accuracy_score', 'f1_score', 'precision_score', 'recall_score'])
                        performance['accuracy_score'] = [accuracy_score(y_test, preds)]
                        performance['f1_score'] = [f1_score(y_test, preds, average='weighted')]
                        performance['precision_score'] = [precision_score(y_test, preds, average='weighted')]
                        performance['recall_score'] = [recall_score(y_test, preds, average='weighted')]

                        st.dataframe(performance, 1000, 50)

                        joblib.dump(RF, RF_file)
                        joblib.dump(cv, cv_new_file)

                except:
                    st.warning('Invalid File...')

    elif choice == 'Similarity':
        st.title('Similarity Problems')
        st.text('Compare your file sequences with Human, Dog and Chimpanzee genes')

        file_similarity = st.file_uploader("Upload fasta file", type=["fas", "fasta", 'txt'])

        if file_similarity is not None:
            human_data = pd.read_table('.\\human_data.txt')
            chimp_data = pd.read_table('.\\chimp_data.txt')
            dog_data = pd.read_table('.\\dog_data.txt')

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

            similarity_human_seq = cosine_similarity(human, cv_seq)
            similarity_dog_seq = cosine_similarity(chimp, cv_seq)
            similarity_chimp_seq = cosine_similarity(dog, cv_seq)
            # np.sort(similarity_human_seq.flatten())[-1:-300:-1]
            similarity_df = pd.DataFrame(columns=['Human', 'Dog', 'Chimpanzee'])

            similarity_df['Human'] = np.sort(similarity_human_seq.flatten())[-1:-300:-1]
            similarity_df['Dog'] = np.sort(similarity_dog_seq.flatten())[-1:-300:-1]
            similarity_df['Chimpanzee'] = np.sort(similarity_chimp_seq.flatten())[-1:-300:-1]

            st.dataframe(similarity_df.head(20), 1000, 500)

            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.bar_chart(similarity_df)

    elif choice == "Alignment":

        seq1 = st.text_input("Enter the 1st Sequence").upper()
        seq2 = st.text_input("Enter the 2nd Sequence").upper()
        # # Finding similarities
        alignments = pairwise2.align.globalxx(seq1, seq2)

        # Showing results

        if st.button("Align"):

            with st.container():
                with st.expander("See Sequence"):
                    for alignment in alignments:
                        st.text(format_alignment(*alignment))


    elif choice == "DNA Mutation":

        st.subheader("DNA Mutation")
        seq_file_mutation = st.file_uploader("Upload FASTA File", type=["fasta", "fa", "fastq"])

        if seq_file_mutation is not None:
            byte_str3 = seq_file_mutation.read()
            text_obj3 = byte_str3.decode('UTF-8')
            dna_record = SeqIO.read(io.StringIO(text_obj3), "fasta")
            # st.write(pep_seq)
            dna_seq = dna_record.seq

            details = st.radio("Details", ("Description", "Sequence"))
            if details == "Description":
                st.write(dna_record.description)
            elif details == "Sequence":
                with st.expander("See Sequence"):
                    st.write(dna_record.seq)
                st.write("--------------------------------------------------------")
                with st.container():
                    if st.checkbox("Insertion Mutation", value=True):
                        mutate_seq = insertion_mutations(dna_record.seq)
                        if st.button("Insertion Plot"):
                            st.write('DNA Sequence:')
                            dna_mute_freq = count_dict(mutate_seq)
                            dna_norm_freq = count_dict(dna_record.seq)
                            visualize_together(dna_norm_freq, dna_mute_freq)

                            st.write('amino acids:')
                            dna_mute_freq = count_dict(mutate_seq.translate())
                            dna_norm_freq = count_dict(dna_record.seq.translate())
                            visualize_together(dna_norm_freq, dna_mute_freq)
                st.write("--------------------------------------------------------")
                with st.container():
                    if st.checkbox("Substitution Mutation", value=True):
                        mutate_seq = substitution_mutations(dna_record.seq)
                        if st.button("Substitution Plot"):
                            st.write('DNA Sequence:')
                            dna_mute_freq = count_dict(mutate_seq)
                            dna_norm_freq = count_dict(dna_record.seq)
                            visualize_together(dna_norm_freq, dna_mute_freq)

                            st.write('amino acids:')
                            dna_mute_freq = count_dict(mutate_seq.translate())
                            dna_norm_freq = count_dict(dna_record.seq.translate())
                            visualize_together(dna_norm_freq, dna_mute_freq)

                st.write("--------------------------------------------------------")
                with st.container():
                    if st.checkbox("Deletion Mutation", value=True):
                        mutate_seq = deletion_mutations(dna_record.seq)
                        if st.button("Deletion Plot"):
                            st.write('DNA Sequence:')
                            dna_mute_freq = count_dict(mutate_seq)
                            dna_norm_freq = count_dict(dna_record.seq)
                            visualize_together(dna_norm_freq, dna_mute_freq)

                            st.write('amino acids:')
                            dna_mute_freq = count_dict(mutate_seq.translate())
                            dna_norm_freq = count_dict(dna_record.seq.translate())
                            visualize_together(dna_norm_freq, dna_mute_freq)


    elif choice == "DNA Sequence":
        st.subheader("DNA Sequence Analysis")

        seq_file = st.file_uploader("Upload FASTA File", type=["fasta", "fa", "fastq"])

        if seq_file is not None:
            byte_str = seq_file.read()
            text_obj = byte_str.decode('UTF-8')
            dna_record = SeqIO.read(io.StringIO(text_obj), "fasta")
            # st.write(pep_seq)
            dna_seq = dna_record.seq

            details = st.radio("Details", ("Description", "Sequence"))
            if details == "Description":
                st.write(dna_record.description)
            elif details == "Sequence":
                with st.expander("See Sequence"):
                    st.write(dna_record.seq)

                # Nucleotide Frequencies
                st.subheader("Nucleotide Frequency")
                dna_freq = Counter(dna_seq)
                with st.expander("See Sequence"):
                    st.write(dna_freq)

                guanine_color = st.color_picker("Guanine Color")
                thymine_color = st.color_picker("ThymineColor")
                adenine_color = st.color_picker("Adenine Color")
                cytosine_color = st.color_picker("Cytosine Color")

                if st.button("Nucleotide Plot"):
                    barlist = plt.bar(dna_freq.keys(), dna_freq.values())
                    barlist[0].set_color(guanine_color)
                    barlist[1].set_color(thymine_color)
                    barlist[2].set_color(adenine_color)
                    barlist[3].set_color(cytosine_color)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                st.subheader("DNA Composition Scores")
                gc_score = utils.gc_content(str(dna_seq))
                at_score = utils.at_content(str(dna_seq))
                st.json({"GC Content": gc_score, "AT Content": at_score})

                # Nucleotide count search
                nt_count = st.text_input("Enter Nucleotide Here: ", "Type Nucleotide Alphabet").upper()
                st.write(("Number of {} Nucleotide is ::{}".format(nt_count, str(dna_seq).count(nt_count))))

                # Protein Synthesis - Transcription - Translation
                st.subheader("Protein Synthesis")
                p1 = dna_seq.translate()
                aa_freq = Counter(str(p1))

                if st.checkbox("Transcription"):
                    transcription = dna_seq.transcribe()
                    with st.expander("See Sequence"):
                        st.write(transcription)
                    st.download_button("Download Transcription", str(transcription),
                                       file_name='seq_transcription.txt')
                if st.checkbox("Translation"):
                    translation = dna_seq.translate()
                    with st.container():
                        with st.expander("See Sequence"):
                            st.write(translation)
                        st.download_button("Download Transcription", str(translation),
                                           file_name='seq_translation.txt')
                if st.checkbox("Complement"):
                    complement = dna_seq.complement()
                    with st.container():
                        with st.expander("See Sequence"):
                            st.write(complement)
                        st.download_button("Download Transcription", str(complement),
                                           file_name='seq_complement.txt')
                if st.checkbox("AA Frequency"):
                    with st.container():
                        with st.expander("See Sequence"):
                            st.write(aa_freq)

                if st.checkbox("AA Plot"):
                    aa_color = st.color_picker("Pick An Amino Acid Color")
                    # barlist = plt.bar(aa_freq.keys(), aa_freq.values())
                    # barlist[0].set_color(aa_color)
                    plt.bar(aa_freq.keys(), aa_freq.values(), color=aa_color)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()

                if st.checkbox("Full Amino Acid Name"):
                    aa_name = str(p1).replace("*", "")
                    aa3 = utils.convert_1to3(aa_name)
                    with st.container():
                        with st.expander("See Sequence"):
                            with st.container():
                                st.write(aa_name)
                        st.write("--------------------------------------------------------")
                        with st.expander("See Sequence"):
                            with st.container():
                                st.write(aa3)

                        st.write("--------------------------------------------------------")
                        get_a = utils.get_acid_name(aa3)
                        with st.expander("See Sequence"):
                            st.write(get_a)

                        st.download_button("Download Full Amino Acid Name", str(get_a), file_name="AA Full Name.txt")

                # Top Most Common Amino


    elif choice == "DotPlot":

        st.subheader("Generate Dot Plot For Two Sequences")
        seq_file1 = st.file_uploader("Upload 1st FASTA File", type=["fasta", "fa", "fastq"])
        seq_file2 = st.file_uploader("Upload 2nd FASTA File", type=["fasta", "fa", "fastq"])

        if seq_file1 and seq_file2 is not None:
            byte_str2 = seq_file1.read()
            text_obj1 = byte_str2.decode('UTF-8')
            dna_record1 = SeqIO.read(io.StringIO(text_obj1), "fasta")
            dna_seq1 = dna_record1.seq

            byte_str3 = seq_file2.read()
            text_obj2 = byte_str3.decode('UTF-8')
            dna_record2 = SeqIO.read(io.StringIO(text_obj2), "fasta")
            dna_seq2 = dna_record2.seq

            details = st.radio("Details", ("Description", "Sequence"))
            if details == "Description":
                st.write(dna_record1.description)
                st.write("--------------------------------------------------------")
                st.write(dna_record2.description)
            elif details == "Sequence":

                with st.expander("See Sequence of sample 1"):
                    st.write(dna_record1.seq)
                st.write("--------------------------------------------------------")

                with st.expander("See Sequence of sample 2"):
                    st.write(dna_record2.seq)

            cus_limit = st.number_input("Select Max number of Nucleotide", 10, 200, 50)
            if st.button("Dot Plot"):
                st.write("Comparing the first {} Nucleotide of the Two Sequences".format(cus_limit))
                dotplotx(dna_seq1[0:cus_limit], dna_seq2[0:cus_limit])
                st.set_option('deprecation.showPyplotGlobalUse', False)

                st.pyplot()

    elif choice == "Blast":
        st.subheader("PSI-BLAST")
        # prot_input = st.text_input("Enter protein sequence")
        html_code = '<a href="https://www.ebi.ac.uk/Tools/sss/psiblast/" target="_blank">GO TO NCBI</a>'
        st.markdown(html_code, unsafe_allow_html=True)
        url_text = st.text_input("Enter URL")

        if st.button("Protein Result"):
            results = requests.get(url_text)

            ID = []
            length = []
            score = []
            identity = []
            positives = []
            E = []

            src = results.content
            soup = BeautifulSoup(src, 'lxml')

            for i in range(1000):
                try:
                    Alignment_results = soup.find_all('tr', {'id': 'alignment_' + str(i)})
                    Alignment = Alignment_results[0].text.split('\n')

                    ID.append(Alignment[6])
                    length.append(Alignment[12])
                    score.append(Alignment[13])
                    identity.append(Alignment[14])
                    positives.append(Alignment[15])
                    E.append(Alignment[16])
                except:
                    pass

            file_lists = [ID, length, score, identity, positives]
            export = zip_longest(*file_lists)

            alignment = open('.\\alignment.csv', 'w+')
            wr = csv.writer(alignment)
            wr.writerow(['ID', 'length', 'score', 'identity', 'positives', 'E'])
            wr.writerows(export)
            st.write(alignment.read())

            st.success('Created Successfully')
            alignment_1 = open('.\\alignment.csv', 'r')
            file = alignment_1.read()
            with st.expander("See result contents"):
                st.text(file)
            st.download_button("Download File", file, file_name="alignment_1.csv")

        if st.button("Visualize Similarity"):
            with open('.\\alignment.csv', 'r') as alignment:
                file = alignment.read()
                file = file.split('\n')
                lines = []
                for i in range(len(file)):
                    if len(file[i]) > 0:
                        lines.append(file[i].split(','))

                score = []
                identity = []
                positive = []
                IDs = []
                for i in range(1, len(lines)):
                    IDs.append(lines[i][0])
                    score.append(lines[i][2])
                    identity.append(lines[i][3])
                    positive.append(lines[i][4])

                st.set_option('deprecation.showPyplotGlobalUse', False)
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        ex = []
                        plt.title('score')
                        for i in range(len(score)):
                            ex.append(0.1)
                        plt.pie(score, labels=IDs, autopct="%2.1f%%", explode=ex)
                        st.pyplot()
                    with col2:
                        ex = []
                        plt.title('identity')
                        for i in range(len(identity)):
                            ex.append(0.1)
                        plt.pie(identity, labels=IDs, autopct="%2.1f%%", explode=ex)
                        st.pyplot()


    elif choice == "About":

        st.subheader("About")
        st.text('''
                        Team Members:
                               1- Eslam Younis
                               2- Steven Waheed
                               3- Maria Sameh 
                               4- Ester elzek
                               5- Mira Rafat
                               6- Silvia Ashraf
                               7- Sandy Esam''')


if __name__ == '__main__':
    main()
