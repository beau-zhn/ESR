from Bio import SeqIO

def get_full_sequence_length_from_cif(cif_file_path):
    """
    Extracts the full biological sequence length (entity sequence) 
    from an mmCIF file. This corresponds to the sequence length on RCSB.

    Args:
        cif_file_path (str): The path to the mmCIF (.cif) file.

    Returns:
        int: The length of the first unique polymer entity sequence, or 0 if none is found.
    """
    try:
        # Use SeqIO to parse the mmCIF file. 
        # The 'mmcif-seqres' format specifically reads the full sequence 
        # defined for each unique polymer entity.
        for seq_record in SeqIO.parse(cif_file_path, "mmcif-seqres"):
            # This returns sequence records based on the unique ENTITY ID.
            # We take the length of the first entity (e.g., Entity 1)
            # which holds the unique protein sequence.
            return len(seq_record)

        return 0 # No polymer entity sequence found

    except FileNotFoundError:
        print(f"Error: File not found at {cif_file_path}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

# --- Example Usage (Using 2BJ4.cif) ---
# You need to download 2BJ4.cif from RCSB PDB
cif_filename = '/Users/aruzhantolegen/Desktop/BIOL363/data set/cif/2BJ4.cif'  # Note the file extension

print(f"Analyzing mmCIF file: {cif_filename} for FULL sequence length")
sequence_length = get_full_sequence_length_from_cif(cif_filename) 

if sequence_length > 0:
    print(f"\n✅ Result: The full biological sequence length is: {sequence_length} residues.")