import sys
import zipfile
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import joblib
from collections import Counter

with tf.device('/GPU:0'):

    OLIGONUCLEOTIDE_MAX_LENGTH = 40
    BATCH_SIZE = 128
    EPOCHS = 5
    PADDING_TARGET_LENGTH = OLIGONUCLEOTIDE_MAX_LENGTH + 8
    PADDING_ELEMENT = [0.25, 0.25, 0.25, 0.25]
    START_PADDING_LEN = 4
    START_PADDING = [PADDING_ELEMENT] * START_PADDING_LEN


    def get_7mers(sequence):
        """Generate all 7-mers from a given DNA sequence."""
        return [sequence[i:i + 7] for i in range(len(sequence) - 6)]


    def get_6mers(sequence):
        """Generate all 6-mers from a given DNA sequence."""
        return [sequence[i:i + 6] for i in range(len(sequence) - 5)]


    def get_5mers(sequence):
        """Generate all 5-mers from a given DNA sequence."""
        return [sequence[i:i + 5] for i in range(len(sequence) - 4)]


    def calculate_sequence_score(func, sequence, kmer_counts):
        """Calculate the average score of 7-mers in a sequence."""
        kmers = func(sequence)
        if kmers:
            scores = [kmer_counts.get(kmer, 0) for kmer in kmers]
            return np.mean(scores)
        return 0


    def get_kmer_counts(df, func):
        # Count n-mers
        all_mers = []
        for seq in df:
            all_mers.extend(func(seq))

        # Calculate n-mer frequencies
        kmer_counts = Counter(all_mers)
        return kmer_counts


    def get_kmer_score(df, rna_sequences):
        """
        Calculate average k-mer scores for a list of RNA sequences based on k-mer counts from a given DataFrame.

        Args:
        - df: DataFrame containing data for k-mer counting.
        - rna_sequences: List of RNA sequences to calculate scores for.

        Returns:
        - A Pandas Series with the average k-mer scores for each RNA sequence.
        """
        kmer_counts = get_kmer_counts(df, get_7mers)
        save_7_scores = [calculate_sequence_score(get_7mers, seq, kmer_counts) for seq in rna_sequences]

        kmer_counts = get_kmer_counts(df, get_6mers)
        save_6_scores = [calculate_sequence_score(get_6mers, seq, kmer_counts) for seq in rna_sequences]

        kmer_counts = get_kmer_counts(df, get_5mers)
        save_5_scores = [calculate_sequence_score(get_5mers, seq, kmer_counts) for seq in rna_sequences]

        RNA_complete = pd.DataFrame({'7_score': save_7_scores, '6_score': save_6_scores, '5_score': save_5_scores})
        save_avg_scores = RNA_complete[['7_score', '6_score', '5_score']].mean(axis=1)
        return save_avg_scores


    def unzip(zip_file, output_dir):
        """
        Extracts the contents of a zip file to a specified output directory.
        """
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)


    def get_cycle_num(filename):

        basename = filename.split('/')[-1]
        name_without_ext = basename.split('.')[0]
        value = name_without_ext.split('_')[-1]
        return int(value)


    def handle_htr_selex_files():
        """
        Processes HTR-SELEX files to extract oligonucleotides and cycle numbers.

        Returns:
        - oligonucleotides_list (list): Extracted and generated oligonucleotides.
        - y_list (list): Corresponding cycle numbers.
        """
        unzip('htr-selex.zip', 'htr-seleblx')
        htr_filenames = sys.argv[2:]

        oligonucleotides_list = []
        y_list = []

        # Actual Cycles
        cycles_amount = len(htr_filenames)
        for i in range(0, cycles_amount):
            cycle_filename = htr_filenames[i]
            try:
                # Try reading from the primary path
                with open(cycle_filename, 'r') as file:
                    lines = file.readlines()
            except FileNotFoundError:
                # If the file is not found, try reading from an alternative path
                alternative_path = "htr-selex/" + htr_filenames[i]
                try:
                    with open(alternative_path, 'r') as file:
                        lines = file.readlines()
                except FileNotFoundError:
                    print(f"File not found in both paths: {cycle_filename} and {alternative_path}")

            cycle = get_cycle_num(cycle_filename)


            for line in lines:
                line_without_n = line.replace('N', '')
                oligonucleotide = line_without_n.split(',')[0]
                oligonucleotides_list.append(oligonucleotide)
                y_list.append(cycle)

        # Cycle 0 (generate oligonucleotides)
        lines_for_cycle_0 = int(len(y_list) / cycles_amount)
        dna_bases = ['A', 'G', 'T', 'C']
        for _ in range(0, lines_for_cycle_0):
            random_string = ''.join(random.choice(dna_bases) for _ in range(40))
            oligonucleotides_list.append(random_string)
            y_list.append(0)

        return oligonucleotides_list, y_list


    def handle_rna_compete_files():
        """
        Processes RNAcompete files to extract intensity values and RNA sequences.

        Returns:
        - intensities (list): List of RNA intensity values.
        - rna_sequences (list): List of RNA sequences.
        """
        unzip('RNAcompete_intensities.zip', 'RNAcompete_intensities')
        rna_filename = sys.argv[1]
        try:
            # Try reading from the primary path
            with open(rna_filename, 'r') as file:
                intensities_lines = file.readlines()
        except FileNotFoundError:
            # If the file is not found, try reading from an alternative path
            alternative_path = "RNAcompete_intensities/" + sys.argv[1]
            try:
                with open(alternative_path, 'r') as file:
                    intensities_lines = file.readlines()
            except FileNotFoundError:
                print(f"File not found in both paths: {rna_filename} and {alternative_path}")

        intensities = [float(num.strip()) for num in intensities_lines]

        rna_compete_seq_filename = "RNAcompete_sequences_rc.txt"
        with open(rna_compete_seq_filename, 'r') as file:
            seq_lines = file.readlines()

        rna_sequences = [line.strip() for line in seq_lines]

        return intensities, rna_sequences


    def one_hot_encode_sequence(sequence):
        """
        Converts a nucleotide sequence to its one-hot encoded representation.
        """
        mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
        return [mapping[nt] for nt in sequence]


    def create_deepselex_model():
        """
        Creates a DeepSELEX neural network model using a convolutional neural network (CNN) architecture.

        Returns:
        - model: A Keras Sequential model for predicting from SELEX data.
        """
        model = models.Sequential()
        model.add(layers.Conv1D(filters=512, kernel_size=8, strides=1, activation='relu',
                                input_shape=(PADDING_TARGET_LENGTH, 4)))

        # Max Pooling layer
        model.add(layers.MaxPooling1D(pool_size=5, strides=5))

        # Fully connected layers
        model.add(layers.Flatten())
        # model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))

        # Output layer
        model.add(layers.Dense(5, activation='softmax'))

        return model


    def train_valid_test_split(X, y):
        """
        Splits data into training, validation, and test sets.
        """
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        X_train, X_val, X_test, y_train, y_val, y_test = (
            np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test))

        return X_train, X_val, X_test, y_train, y_val, y_test


    def main():

        oligonucleotides_list, cycles_list = handle_htr_selex_files()
        #rna_intensities, rna_sequences = handle_rna_compete_files()       # For training usage

        # ------------------------------- MODEL -------------------------------
        encoded_sequences = [one_hot_encode_sequence(seq) for seq in oligonucleotides_list]

        padded_sequences = [np.concatenate(
            [START_PADDING, arr, [PADDING_ELEMENT] * (PADDING_TARGET_LENGTH - len(arr) - START_PADDING_LEN)])
            for arr in encoded_sequences]

        X_train, X_val, X_test, y_train, y_val, y_test = train_valid_test_split(X=padded_sequences, y=cycles_list)

        model = create_deepselex_model()
        adam_optimizer = optimizers.Adam(learning_rate=0.005, beta_1=0.85, beta_2=0.95)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))

        rna_compete_seq_filename = sys.argv[1]
        with open(rna_compete_seq_filename, 'r') as file:
            seq_lines = file.readlines()

        rna_sequences = [line.strip() for line in seq_lines]

        rc_encoded_sequences = [one_hot_encode_sequence(seq) for seq in rna_sequences]
        rc_padded_sequences = [np.concatenate(
            [START_PADDING, arr, [PADDING_ELEMENT] * (PADDING_TARGET_LENGTH - len(arr) - START_PADDING_LEN)])
            for arr in rc_encoded_sequences]

        rc_padded_sequences = np.array(rc_padded_sequences)
        rna_intensities_preds = model.predict(rc_padded_sequences)

        # ------------------------------- STATISTICAL APPROACH -------------------------------
        filtered_oligonucleotides_3_or_4 = [oligo for oligo, cycle in zip(oligonucleotides_list, cycles_list) if
                                            cycle in [3, 4]]
        avg_stat_scores = get_kmer_score(filtered_oligonucleotides_3_or_4, rna_sequences)
        normalized_kmer_score = 4 * (avg_stat_scores - avg_stat_scores.min()) / (
                avg_stat_scores.max() - avg_stat_scores.min())

        # ------------------------------- INTENSITIES -------------------------------
        # XGBOOST
        # For training usage
        # xgb_model = XGBRegressor(learning_rate=0.01, max_depth=15)
        # xgb_model.fit(rna_intensities_preds_train, rna_intensities_train)
        # xgb_predictions = xgb_model.predict(rna_intensities_preds_test)
        # joblib.dump(xgb_model, f"XGB_model_{rbp_num}.pkl")

        # Iterate through all the pkl files calculated in the training process and calculate the average prediction value
        model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if len(model_files) == 0:
            intensities_predictions = np.array(normalized_kmer_score)

        else:
            # Initialize list to hold models
            models = []

            # Load all models
            for model_file in model_files:
                model = joblib.load(model_file)
                models.append(model)

            # Calculate the weight for each model
            n_models = len(models)
            weights = [1 / n_models] * n_models

            # Get predictions from all models
            model_predictions = [model.predict(rna_intensities_preds) for model in models]

            # Stack predictions into an array with shape (num_samples, num_models)
            predictions_array = np.stack(model_predictions, axis=1)

            # Blend predictions
            blended_predictions = np.dot(predictions_array, weights)
            intensities_predictions = (np.array(normalized_kmer_score) + np.array(blended_predictions)) / 2

        intensities_predictions_str = [str(num) + '\n' for num in intensities_predictions]
        rbp_file_name = rna_compete_seq_filename.replace(".txt", "_predictions.txt")
        with open(rbp_file_name, 'w') as file:
            file.writelines(intensities_predictions_str)



    if __name__ == "__main__":
        main()
