import sys

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        return

    rna_filename = "RNAcompete_intensities/" + sys.argv[1]
    htr_filenames = sys.argv[2:]

    # Create a dictionary to store HTR-SELEX filenames
    htr_folder = "htr-selex/"
    htr_files = {f"htr_filename{i + 1}": htr_folder + htr_filenames[i] for i in range(len(htr_filenames))}

    # # Print contents of each HTR-SELEX file with prefix
    # for idx, (key, htr_filename) in enumerate(htr_files.items(), start=1):
    #     print(f"\nContents of HTR-SELEX file {idx} ({key} - {htr_filename}):")
    #     try:
    #         with open(htr_filename, 'r') as file:
    #             print(file.read())
    #     except FileNotFoundError:
    #         print(f"Error: File '{htr_filename}' not found.")


if __name__ == "__main__":
    main()
