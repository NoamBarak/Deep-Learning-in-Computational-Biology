import sys
import zipfile
import os

def unzip(zip_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)     # Create output directory if it doesn't exist

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        return

    # Unzip
    unzip('htr-selex.zip', 'htr-selex')
    unzip('RNAcompete_intensities.zip', 'RNAcompete_intensities')


    # Input files
    rna_filename = "RNAcompete_intensities/" + sys.argv[1]
    htr_filenames = sys.argv[2:]
    htr_files = {f"htr_filename{i + 1}": "htr-selex/" + htr_filenames[i] for i in range(len(htr_filenames))}




if __name__ == "__main__":
    main()
