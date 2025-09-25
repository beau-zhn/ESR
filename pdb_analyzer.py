from Bio import SeqIO

def get_full_fasta_sequence_length(pdb_file_path):
    """
    Extracts the full biological sequence length (SEQRES) from a PDB file.

    Args:
        pdb_file_path (str): The path to the PDB file.

    Returns:
        int: The length of the first unique sequence record, or None if an error.
    """
    try:
        # Use SeqIO to parse the PDB file format. 
        # This automatically reads the SEQRES records.
        for seq_record in SeqIO.parse(pdb_file_path, "pdb-seqres"):
            # The 'pdb-seqres' format yields records for each unique entity.
            # We return the length of the first unique entity found.
            return len(seq_record)

        return 0 # Should only happen if no SEQRES record is present

    except FileNotFoundError:
        print(f"Error: File not found at {pdb_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Rerunning Example ---
pdb_filename = '/Users/aruzhantolegen/Desktop/BIOL363/data set/PDB/3OS8.pdb'  

print(f"Analyzing PDB file: {pdb_filename} for FULL sequence length (SEQRES)")
sequence_length = get_full_fasta_sequence_length(pdb_filename) 

if sequence_length is not None:
    print(f"\n✅ Result: The full biological sequence length (SEQRES) is: {sequence_length} residues.")