"""
This file runs Q5.
Note: The symbols and messages text file and this code file should be in the same folder level.
"""
# -------------------------------------------------------------------------------------------------------------------- #
# Standard Library
from typing import Dict, Tuple, Optional, List
import collections
import re
# Third Party
from scipy.special import expit
import seaborn as sns
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


# Private


# -------------------------------------------------------------------------------------------------------------------- #
class Cipher:

    @staticmethod
    def clean_text(reference_text: str, symbol_text: str):
        """
        Given a text, clean the document such that all the symbols we used for encryption appear in the reference text.
        Args:
            reference_text:
                String to reference the decrypted text cipher against for likelihood scoring.
            symbol_text:
                The string name of the file containing symbols that were used for encryption.
        Returns:
            A cleaned string version.
        """
        # Symbols used for mapping
        symbol_list = list(symbol_text)
        # Check the unique letters within reference text
        reference_unique_char = sorted(set(reference_text))
        # Handle unknown characters that do not match symbol file
        unknown_symbols = []
        for char in reference_unique_char:
            if char not in symbol_list:
                unknown_symbols.append(char)
        # Remove symbols that are not in the symbol file
        reference_text = re.sub(r'[\n#$%@]', '', reference_text)
        # Change to lower case vowels
        reference_text = re.sub(r'À', "a", reference_text)
        reference_text = re.sub(r'Á', "a", reference_text)
        reference_text = re.sub(r'É', "e", reference_text)
        reference_text = re.sub(r'à', "a", reference_text)
        reference_text = re.sub(r'á', "a", reference_text)
        reference_text = re.sub(r'ä', "a", reference_text)
        reference_text = re.sub(r'â', "a", reference_text)
        reference_text = re.sub(r'è', "e", reference_text)
        reference_text = re.sub(r'é', "e", reference_text)
        reference_text = re.sub(r'ê', "e", reference_text)
        reference_text = re.sub(r'ë', "e", reference_text)
        reference_text = re.sub(r'í', "i", reference_text)
        reference_text = re.sub(r'î', "i", reference_text)
        reference_text = re.sub(r'ï', "i", reference_text)
        reference_text = re.sub(r'ó', "o", reference_text)
        reference_text = re.sub(r'ô', "o", reference_text)
        reference_text = re.sub(r'ö', "o", reference_text)
        reference_text = re.sub(r'ú', "u", reference_text)
        reference_text = re.sub(r'ü', "u", reference_text)
        # Fix digraphs
        reference_text = re.sub(r'æ', "ae", reference_text)
        reference_text = re.sub(r'œ', "oe", reference_text)
        # Change to lower case consonants
        reference_text = re.sub(r'ç', "c", reference_text)
        reference_text = re.sub(r'ý', "y", reference_text)
        # Change others to match symbols in symbol file
        reference_text = re.sub(r'—', "-", reference_text)
        reference_text = re.sub(r'‘', symbol_list[10], reference_text)
        reference_text = re.sub(r'’', symbol_list[10], reference_text)
        reference_text = re.sub(r'“', symbol_list[11], reference_text)
        reference_text = re.sub(r'”', symbol_list[11], reference_text)
        reference_text = re.sub(r'\ufeff', symbol_list[1], reference_text)
        # Finally make text all lower case
        reference_text = reference_text.lower()
        # Check one more time to see if any unknown symbols in reference text do not appear in symbol file
        reference_unique_char_final = sorted(set(reference_text))
        unknown_symbols_final = []
        for char in reference_unique_char_final:
            if char not in symbol_list:
                unknown_symbols_final.append(char)
        if len(unknown_symbols_final) != 0:
            print("There are still some symbols that have not been converted! Do not include these in transition"
                  "matrix.")
        return reference_text

    @staticmethod
    def cipher_map(symbol_text: str, symbol_counts: Dict, reference_counts: Dict, random_cipher: bool) -> Dict:
        """
        Given a file of symbols, return a dictionary in the form {A:R, B:W, C:O, ...}.
        Args:
            symbol_text:
                The string name of the file containing symbols that were used for encryption.
            symbol_counts:
                Dictionary containing symbol counts in symbol file.
            reference_counts:
                Dictionary containing symbol counts in reference text.
            random_cipher:
                Whether the starting cipher is random or chosen with prior knowledge.
        Returns:
            A dictionary containing the map from symbol (encryption key) to encoded key (encrypted text).
        """
        # Create random cipher
        if random_cipher is False:
            new_cipher = ''.join(random.sample(symbol_text, len(symbol_text)))
            # Create map
            mapping = {}
            for i in range(len(new_cipher)):
                mapping[new_cipher[i]] = symbol_text[i]
            return mapping
        # Create optimal cipher from matching highest counts frequency distribution
        else:
            mapping = {}
            all_symbols = list(symbol_text)
            symbol_counts_ordered = {k: v for k, v in sorted(symbol_counts.items(), key=lambda item: item[1],
                                                             reverse=True)}
            reference_counts_ordered = {k: v for k, v in sorted(reference_counts.items(), key=lambda item: item[1],
                                                                reverse=True)}
            for key_1, key_2 in zip(list(symbol_counts_ordered.keys()), list(reference_counts_ordered.keys())):
                mapping[key_1] = key_2
                all_symbols.pop(all_symbols.index(key_1))
            # Find remaining keys that have not been mapped and map them randomly
            random.shuffle(all_symbols)
            for final_keys in all_symbols:
                mapping[final_keys] = final_keys
            return mapping

    @staticmethod
    def apply_cipher(text: str, cipher_map: Dict) -> str:
        """
        Given a text, apply the cipher map to the text to decrypt it.
        Args:
            text:
                String to encrypt.
            cipher_map:
                The map illustrating how the encrypted text is mapping to the original symbols.
        Returns:
            String of text.
        """
        # Apply cipher on text
        new_text = ''
        for i in range(len(text)):
            new_text += cipher_map[text[i]]
        return new_text

    def generate_new_cipher(self, cipher_map: Dict) -> Dict:
        """
        Given a cipher map, create a new cipher via a proposal, which switches two symbol permutation outputs.
        Args:
            cipher_map:
                The cipher map.
        Returns:
            A new cipher map.
        """
        # Create copy of original cipher
        cipher_map_new = cipher_map.copy()
        # Pick at random two symbols
        first_swap = np.random.choice(list(cipher_map.keys()))
        second_swap = np.random.choice(list(cipher_map.keys()))
        # Make sure random choice does not pick the same symbol
        while first_swap == second_swap:
            first_swap = np.random.choice(list(cipher_map.keys()))
        # Swap the corresponding encrypted symbols of these picks
        cipher_map_new[first_swap] = cipher_map[second_swap]
        cipher_map_new[second_swap] = cipher_map[first_swap]
        # Make sure new cipher has not just swapped back previous cipher
        while cipher_map == cipher_map_new:
            cipher_map_new = self.generate_new_cipher(cipher_map=cipher_map)
        return cipher_map_new

    @staticmethod
    def cipher_score(decrypted_text: str,
                     reference_text_transition_matrix: pd.DataFrame,
                     decrypted_single_start: float) -> float:
        """
        Given a cipher, calculate its score function.
        Args:
            decrypted_text:
                The decrypted text to calculate likelihood on.
            reference_text_transition_matrix:
                The transition probabilities of each pair (in this case we use War and Peace).
            decrypted_single_start:
                The probability that of starting value of the decrypted text.
        Returns:
            A float value.
        """
        # Calculate negative log likelihood score
        score = 0
        for i, j in zip(decrypted_text, decrypted_text[1:]):
            score += np.log(reference_text_transition_matrix[i][j])
        score += np.log(decrypted_single_start)
        return score


class MCMC(Cipher):

    def __init__(self, symbol_text: str, reference_text: str, encrypted_text: str, start_key: Optional[str]):
        """
        Initialise MCMC algorithm.
        Args:
            symbol_text:
                String containing symbols that were used in encryption.
            reference_text:
                String to reference the decrypted text cipher against for likelihood scoring.
            encrypted_text:
                The encrypted text in string form.
            start_key:
                The starting decryption key to begin the Markov chain.
        """
        self.symbol_text = symbol_text
        self.reference_text = reference_text
        self.encrypted_text = encrypted_text
        self.start_key = start_key

    @staticmethod
    def get_single_counts_probas(text: str) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Given a text file, obtain the number of occurrences of each character and thus the probability.
        This is calculating P(s).
        Args:
            text:
                The string of the text.
        Returns:
            A dictionary containing the character and its corresponding count, with a dataframe.
        """
        # Creating dictionary
        text_counts = dict(collections.Counter(text))
        # Creating probabilities
        text_probas_normalised = {k: v / total for total in (sum(text_counts.values()),
                                                             ) for k, v in text_counts.items()}
        return text_counts, text_probas_normalised

    def get_pair_counts_and_probas(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Given a text file, obtain the transition probabilities between character and successive character.
        This is calculating P(s_i=α|s_i-1=β).
        Args:
            N/A
        Returns:
            A square matrix of transition probabilities over all characters and their counts.
        """
        # Obtain dictionary containing available characters in file
        symbol_list = sorted(self.symbol_text)
        # Calculate transition probabilities
        final_counts = {}
        pair_counts = {}
        # Loop through each symbol over every other symbol
        transition_matrix = pd.DataFrame(data={}, index=sorted(self.symbol_text), columns=sorted(self.symbol_text))
        for i in symbol_list:
            for j in symbol_list:
                # Store number of counts (Note: To fix empty transition, add 1 - this does not affect chain)
                pair_counts[f"{i}{j}"] = self.reference_text.count(f"{i}{j}")
            # Normalise transition probabilities
            transition_matrix[i] = [v / total for total in (sum(pair_counts.values()),) for _,
                                                                                            v in pair_counts.items()]
            final_counts.update(pair_counts)
            # Reinitialise pair counts
            pair_counts = {}
        # If any transitions are 0, add the minimum value of the transition matrix to every value and normalise again
        min_value = np.amin(transition_matrix[transition_matrix != 0].fillna(1).to_numpy())
        transition_matrix += min_value
        return final_counts, transition_matrix

    @staticmethod
    def acceptance_criteria(old_cipher_score: float, new_cipher_score: float) -> bool:
        """
        Given a cipher and its successor, decide whether to accept the new cipher.
        Args:
            old_cipher_score:
                Score of the old cipher.
            new_cipher_score:
                Score of the new cipher.
        Returns:
            Boolean - if True, accept new cipher, else reject new cipher and try again.
        """
        # Define rejection kernel
        A = np.min([1, expit(new_cipher_score - old_cipher_score)])
        # Create random uniform
        acceptance_proba = np.random.uniform(0, 1)
        # Execute proposal
        if acceptance_proba <= A:
            # Accept proposal
            return True
        else:
            return False

    def run_alg(self, num_iter: int, random_cipher: bool) -> Dict[str, List]:
        """
        Run MCMC (via Metropolis Hastings) algorithm.
        Args:
            num_iter:
                The number of iterations to run the Markov chain.
            random_cipher:
                Whether the starting cipher is random or chosen with prior knowledge.
        Returns:
            Decoded text and final cipher map.
        """
        print("MCMC algorithm warmup...")
        # Obtain single and pair transition probabilities from reference text as well as counts
        reference_single_text = self.get_single_counts_probas(text=self.reference_text)
        reference_counts = reference_single_text[0]
        # Calculate stationary distribution of reference text
        reference_single_text_probas = reference_single_text[1]
        reference_text_transition_matrix = self.get_pair_counts_and_probas()[1]
        symbol_counts = self.get_single_counts_probas(text=self.encrypted_text)[0]
        # Store ciphers, likelihood score and decrypted texts
        best_state = {}
        # Create starting cipher decryption map i.e. the identity map
        cipher_map = self.cipher_map(symbol_text=self.symbol_text,
                                     symbol_counts=symbol_counts,
                                     reference_counts=reference_counts,
                                     random_cipher=random_cipher,
                                     )
        # Using this cipher, decrypt the encrypted text
        decrypted_text = self.apply_cipher(text=self.encrypted_text, cipher_map=cipher_map)
        # Calculate first prior probability
        decrypted_single_start = reference_single_text_probas[decrypted_text[0]]
        # Calculate score of decrypted text
        old_cipher_score = self.cipher_score(decrypted_text=decrypted_text[1:],
                                             reference_text_transition_matrix=reference_text_transition_matrix,
                                             decrypted_single_start=decrypted_single_start)
        # Add starting cipher to best state
        best_state.update({old_cipher_score: cipher_map})
        print("MCMC commencing: \n")
        for i in range(num_iter):
            # Generate new cipher
            new_cipher_map = self.generate_new_cipher(cipher_map=cipher_map)
            while new_cipher_map in list(best_state.values()):
                new_cipher_map = self.generate_new_cipher(cipher_map=cipher_map)
            # Decrypt the encrypted text with this new cipher
            new_decrypted_text = self.apply_cipher(text=self.encrypted_text, cipher_map=new_cipher_map)
            # Print decrypted text every 100 iterations
            if i % 100 == 0:
                print(f"Likelihood: {old_cipher_score} | First 60 strings from decrypted text at iteration {i}: "
                      f"{new_decrypted_text[:60]}.")
            # Calculate pair counts in new decrypted text
            decrypted_single_start = reference_single_text_probas[decrypted_text[0]]
            # Calculate score of new cipher
            new_cipher_score = self.cipher_score(decrypted_text=new_decrypted_text[1:],
                                                 reference_text_transition_matrix=reference_text_transition_matrix,
                                                 decrypted_single_start=decrypted_single_start)
            # Execute acceptance proposal
            if self.acceptance_criteria(old_cipher_score=old_cipher_score, new_cipher_score=new_cipher_score) is True:
                # Store new cipher and likelihood score
                best_state.update({new_cipher_score: [new_cipher_map, new_decrypted_text]})
                # Update old cipher map with new cipher map
                cipher_map = new_cipher_map
                # Update old cipher score with new cipher score
                old_cipher_score = new_cipher_score
        # Return best cipher (Note: The best state is not necessarily increasing in score due to acceptance criteria)
        return self.best_configuration(state_list=best_state)

    @staticmethod
    def best_configuration(state_list: Dict):
        """
        Given a dictionary, with the key as a likelihood value and value as cipher map, find the best cipher.
        Args:
            state_list:
                A list of ciphers and their likelihood values.
        Returns:
            A dictionary, containing the likelihood value and best cipher
        """
        best_configuration = np.argmax(list(state_list.keys()))
        best_score = list(state_list.keys())[best_configuration]
        best_map = list(state_list.values())[best_configuration][0]
        best_decrypted_text = list(state_list.values())[best_configuration][1]
        return {best_score: [best_map, best_decrypted_text]}

    @staticmethod
    def calculate_stationary_distribution(transition_matrix: np.array):
        """
        Given a transition matrix, calculate its stationary distribution.
        Args:
            transition_matrix:
                A matrix of transition probabilities.
        Returns:
            An array of probabilities.
        """
        # Initialise uniform
        p_uniform = np.random.uniform(0, 1, transition_matrix.shape[1]).reshape(1, -1)
        p_uniform = p_uniform / p_uniform.sum()
        # Loop
        for _ in range(10000):
            # Calculate next iteration
            p_new = p_uniform @ transition_matrix
            # Find norm
            if np.linalg.norm(p_new-p_uniform) < 1e-5:
                break
            else:
                p_uniform = p_new
        return p_uniform

    @staticmethod
    def plot_transition_matrix(transition_matrix: pd.DataFrame):
        """
        Given a transition matrix, plot the dataframe.
        Args:
            transition_matrix:
                Transition matrix of probabilities.
        Returns:
            Seaborn heatmap plot.
        """
        plt.figure(figsize=(20, 20))
        sns.set(font_scale=2)
        sns.heatmap(transition_matrix, vmin=0, vmax=1, xticklabels=list(transition_matrix.index),
                    yticklabels=list(transition_matrix.index), cmap="rocket_r")
        plt.show()

    @staticmethod
    def plot_stationary_distribution(stationary_distribution: Dict):
        """
        Given stationary counts, plot the stationary distribution as a dataframe.
        Args:
            stationary_distribution:
                Vector of probabilities.
        Returns:
            Seaborn heatmap plot.
        """
        plt.figure(figsize=(8, 20))
        sns.set(font_scale=2)
        values = list(stationary_distribution.values())
        key = list(stationary_distribution.keys())
        sns.heatmap(np.array(values).reshape(-1, 1), vmin=0, vmax=1, yticklabels=key, xticklabels=["_"],
                    cmap="rocket_r")
        plt.show()


# Create main function to run via terminal
def main():
    # Obtain text files
    symbol_text = open("symbols.txt").read().replace('\n', '')
    encrypted_text = open("message.txt").read()
    with open('war_and_peace.txt') as f:
        reference_text = f.read()
    # Clean reference text of characters
    reference_text = Cipher.clean_text(reference_text=reference_text, symbol_text=symbol_text)
    # Instantiate Metropolis Hastings Algorithm
    decryption_model = MCMC(symbol_text=symbol_text, reference_text=reference_text, encrypted_text=encrypted_text,
                            start_key=symbol_text)
    print("MCMC algorithm initialised!")
    # Run Metropolis Hastings algorithm
    print("Enter number of iterations to run MCMC Algorithm:")
    num_iter = int(input())
    print("Enter True for a specialised cipher, else enter False:")
    random_cipher = eval(input())
    cipher_list = decryption_model.run_alg(num_iter=num_iter, random_cipher=random_cipher)
    print("Final Results:\n")
    print(f"Best Score: {list(cipher_list.keys())[0]}")
    print(f"Best Map: {list(cipher_list.values())[0][0]}")
    print(f"Decrypted Text: {list(cipher_list.values())[0][1]}")


# Execute code via terminal
if __name__ == "__main__":
    main()
