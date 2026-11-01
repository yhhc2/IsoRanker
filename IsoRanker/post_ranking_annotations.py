import os
import pysam
import pandas as pd
import numpy as np
import re
from pyhpo import stats, Ontology, HPOSet
from pathlib import Path


def merge_tsvs_by_keyword(directory, keyword, output_tsv):
    """
    Merges tsv files in the specified directory that contain a given keyword in their filename.
    The function preserves column order from the first file and includes all columns across files.
    Adds a column `Source_File` to track the original file for each row.

    Parameters:
    - directory (str): Path to the directory containing tsv files.
    - keyword (str): Keyword to match files (e.g., "gene" or "isoform").
    - output_tsv (str): Output file path for the merged tsv.

    Returns:
    - None: Saves the merged tsv to the specified output path.
    """

    tsv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".tsv.gz") and keyword.lower() in f.lower()]

    if not tsv_files:
        print(f"No matching files found for keyword '{keyword}'. Skipping...")
        return

    first_df = pd.read_csv(tsv_files[0], compression = "gzip", sep="\t")
    all_columns = list(first_df.columns)
    all_columns.append("Source_File")

    for file in tsv_files[1:]:
        df = pd.read_csv(file, compression = "gzip", sep="\t")
        new_columns = [col for col in df.columns if col not in all_columns]
        all_columns.extend(new_columns)

    merged_df = pd.concat(
        [pd.read_csv(f, compression = "gzip", sep="\t").reindex(columns=all_columns).assign(Source_File=os.path.basename(f)) for f in tsv_files],
        ignore_index=True
    )

    merged_df.to_csv(output_tsv, index=False, compression = "gzip", sep="\t")
    print(f"Merged {len(tsv_files)} '{keyword}' files into: {output_tsv}")



# I think this is deprecated
def merge_haplotype_data(sample_gene_file, haplotype_patient_file, output_file):
    """
    Merges haplotype data into a master file containing Sample and associated_gene.

    Parameters:
    - sample_gene_file (str): Path to the tsv file containing 'Sample' and 'associated_gene' columns.
    - haplotype_patient_file (str): Path to the tsv file containing 'Gene', 'Hap1', 'Hap2', 'Hap0', and 'Patient'.
    - output_file (str): Path to save the merged tsv.

    Returns:
    - None: Saves the merged tsv to the specified output file.
    """

    sample_gene_df = pd.read_csv(sample_gene_file, sep="\t")
    haplotype_df = pd.read_csv(haplotype_patient_file, sep="\t")

    merged_df = sample_gene_df.merge(
        haplotype_df,
        left_on=["Sample", "associated_gene"],
        right_on=["Patient", "Gene"],
        how="left"
    ).drop(columns=["Patient", "Gene"], errors="ignore")

    merged_df.to_csv(output_file, index=False, sep="\t")
    print(f"Merged data saved to: {output_file}")


def process_phenotype_data(hpo_file, genemap_file, probands_file, output_prefix="all_proband_by_omim_comparison_ic"):
    """
    Processes phenotype, gene, and disease data to compute HPO similarity scores between probands and OMIM diseases.

    Parameters:
    hpo_file (str): Path to the phenotype.hpoa file.
    genemap_file (str): Path to the genemap2 file.
    probands_file (str): Path to the probands file containing 'Sample' and 'Phenotype' columns.
    output_prefix (str): Prefix for output files. Default is 'all_proband_by_omim_comparison_ic'.

    Outputs:
    - all_proband_by_omim_comparison_ic.tsv
    - all_proband_by_omim_comparison_ic_longFormat.tsv
    """

    # Load HPO annotations
    hpo = pd.read_csv(hpo_file, sep='\t', comment='#')
    hpo = hpo[hpo['aspect'].isin(['P', 'H'])][['database_id', 'hpo_id']].drop_duplicates()
    hpo_dict = hpo.groupby('database_id')['hpo_id'].apply(lambda x: x.unique().tolist()).to_dict()

    # Load and process probands data
    probands = pd.read_csv(probands_file, sep="\t")[['Sample', 'Phenotype']].drop_duplicates()
    probands = probands[probands['Phenotype'].notnull()]

    Ontology()  # Load HPO Ontology

    def string_to_hpo(input_list, silent=False):
        """Convert phenotype terms to HPO terms, handling unknown terms gracefully."""
        output_list = []
        for term in input_list:
            try:
                hpo_term = str(Ontology.get_hpo_object(term.strip())).split(" | ")[0]
                output_list.append(hpo_term)
            except RuntimeError:
                if not silent:
                    print(f"Warning: Proband HPO term not found for '{term.strip()}'", flush=True)
        return output_list if output_list else np.nan  # Return NaN if no terms were found

    probands['hpo_terms'] = probands['Phenotype'].apply(lambda x: string_to_hpo(x.strip().split(",")))
    probands['hpo_set'] = probands['hpo_terms'].apply(HPOSet.from_queries)

    # Load and process genemap file
    genemap = pd.read_csv(genemap_file, sep='\t', skiprows=3)  # Read the file, skipping the first 3 rows

    # Extract phenotype details from the 'Phenotypes' column
    def extract_phenotype_details(phenotype):
        """Extract phenotype details using regex."""
        phenotype_pattern = r'^(?:\{|\[)?(.*?)(?:\}|\])?, (\d{6}) \((\d+)\)(?:, )?(.*)?$'
        match = re.match(phenotype_pattern, phenotype)
        
        if match:
            phenotype_name, omim_num, code, inheritance = match.groups()
        else:
            phenotype_name, omim_num, code, inheritance = np.nan, np.nan, np.nan, np.nan

        if inheritance == "":
            inheritance = np.nan
        return phenotype_name, omim_num, code, inheritance

    new_rows = []
    for _, row in genemap.iterrows():
        phenotypes = row['Phenotypes']

        if pd.isna(phenotypes):
            new_row = row.to_dict()
            new_row.update({
                'PhenotypesExtracted_PhenotypeName': np.nan,
                'PhenotypesExtracted_OMIMnum': np.nan,
                'PhenotypesExtracted_code': np.nan,
                'PhenotypesExtracted_inheritance': np.nan
            })
            new_rows.append(new_row)
        else:
            for phenotype in phenotypes.split(';'):
                phenotype = phenotype.strip()
                phenotype_name, omim_num, code, inheritance = extract_phenotype_details(phenotype)
                new_row = row.to_dict()
                new_row.update({
                    'PhenotypesExtracted_PhenotypeName': phenotype_name,
                    'PhenotypesExtracted_OMIMnum': omim_num,
                    'PhenotypesExtracted_code': code,
                    'PhenotypesExtracted_inheritance': inheritance
                })
                new_rows.append(new_row)

    # Create a new DataFrame with extracted phenotype details
    genemap = pd.DataFrame(new_rows)

    # Keep only genes with known molecular basis (code 3)
    genemap = genemap[genemap["PhenotypesExtracted_code"] == "3"]
    genemap = genemap[genemap['Approved Gene Symbol'].notnull()]
    genemap = genemap[['Approved Gene Symbol', 'PhenotypesExtracted_OMIMnum', 'PhenotypesExtracted_code']].drop_duplicates()

    genemap['PhenotypesExtracted_OMIMnum'] = genemap['PhenotypesExtracted_OMIMnum'].apply(
        lambda x: f"OMIM:{int(x)}" if pd.notna(x) else x
    )
    genemap['HPO_terms'] = genemap['PhenotypesExtracted_OMIMnum'].map(hpo_dict).fillna(np.nan)

    def filter_valid_hpo_terms(hpo_terms):
        """Filters out invalid HPO terms before creating an HPOSet, logging invalid terms."""
        if not isinstance(hpo_terms, list) or len(hpo_terms) == 0:
            return np.nan  # Return NaN if empty or invalid
        
        valid_terms = []
        invalid_terms = []
    
        for term in hpo_terms:
            try:
                if Ontology.get_hpo_object(term):  # Ensure term exists
                    valid_terms.append(term)
            except RuntimeError:
                invalid_terms.append(term)  # Log invalid terms
    
        # Print invalid terms if any
        if invalid_terms:
            print(f"Warning: Invalid genemap HPO terms found and ignored: {invalid_terms}", flush=True)
    
        return HPOSet.from_queries(valid_terms) if valid_terms else np.nan  # Return NaN if no valid terms

    # Apply the function safely
    genemap['HPO_set'] = genemap['HPO_terms'].apply(filter_valid_hpo_terms)

    # Convert HPOSet to a string for deduplication
    genemap['HPO_set_str'] = genemap['HPO_set'].apply(lambda x: str(x) if isinstance(x, HPOSet) else np.nan)

    # Drop duplicates using both OMIM number and HPO_set_str
    genemap_by_omim = genemap[['PhenotypesExtracted_OMIMnum', 'HPO_set', 'HPO_set_str']].drop_duplicates(subset=['PhenotypesExtracted_OMIMnum', 'HPO_set_str'])


    # Compute similarity scores
    all_comparisons = pd.DataFrame(index=genemap_by_omim['PhenotypesExtracted_OMIMnum'], columns=probands['Sample'])

    for o in genemap_by_omim.itertuples():
        for p in probands.itertuples():
            p_set = p.hpo_set
            o_set = o.HPO_set
            similarity = p_set.similarity(o_set, method='ic') if pd.notna(o_set) else np.nan
            all_comparisons.at[o.PhenotypesExtracted_OMIMnum, p.Sample] = similarity

    all_comparisons_long = all_comparisons.reset_index().melt(
        id_vars=['PhenotypesExtracted_OMIMnum'], var_name='Sample', value_name='similarity'
    )

    # Convert OMIM-associated HPO terms to phenotype strings
    def hpo_codes_to_strings(hpo_terms):
        """
        Convert a list of HPO codes to their human-readable phenotype descriptions.
        
        Parameters:
        hpo_terms (list): List of HPO term codes (e.g., ['HP:0001250', 'HP:0004322']).
        
        Returns:
        list: List of human-readable HPO phenotype descriptions.
        """
        if not isinstance(hpo_terms, list):
            return np.nan  # Return NaN if not a list
        
        phenotype_strings = []
        for hpo_code in hpo_terms:
            try:
                hpo_object = Ontology.get_hpo_object(hpo_code.strip())
                phenotype_strings.append(hpo_object.name)  # Extract human-readable name
            except RuntimeError:
                continue  # Skip invalid terms
        
        return phenotype_strings if phenotype_strings else np.nan  # Return NaN if empty

    # **Add Gene Name, Sample HPO Terms, and OMIM HPO Terms to `all_comparisons_long`**
    genemap['HPO_terms_OMIM'] = genemap['HPO_terms'].apply(hpo_codes_to_strings)
    probands['HPO_terms_Sample'] = probands['hpo_terms'].apply(hpo_codes_to_strings)
    all_comparisons_long = all_comparisons_long.merge(
        genemap[['PhenotypesExtracted_OMIMnum', 'Approved Gene Symbol', 'HPO_terms_OMIM']],
        on='PhenotypesExtracted_OMIMnum',
        how='left'
    ).merge(
        probands[['Sample', 'HPO_terms_Sample']],
        on='Sample',
        how='left'
    )

    # Save results
    all_comparisons.to_csv(f"{output_prefix}.tsv.gz", compression = "gzip", sep="\t")
    all_comparisons_long.to_csv(f"{output_prefix}_longFormat.tsv.gz", index=False, compression = "gzip", sep="\t")

    return all_comparisons, all_comparisons_long


def split_fusion_genes(df, gene_col="associated_gene", preserve_col="original_associated_gene"):
    """
    Splits rows where the gene_col contains an underscore (excluding 'novel'),
    duplicating rows for each gene while preserving other column values.
    
    Parameters:
    - df: pandas DataFrame
    - gene_col: column name with potentially fused gene names (default: 'associated_gene')
    - preserve_col: column name to store the original fused gene string
    
    Returns:
    - Expanded DataFrame with split genes
    """
    df = df.copy()
    
    # Only add preserve_col if it doesn't already exist
    if preserve_col not in df.columns:
        df[preserve_col] = df[gene_col]
    
    # Identify rows to split (fusion genes, not containing 'novel')
    mask = df[gene_col].str.contains("_") & ~df[gene_col].str.contains("novel", case=False, na=False)
    rows_to_split = df[mask]
    rows_to_keep = df[~mask]
    
    # Split and explode
    split_rows = rows_to_split.copy()
    split_rows[gene_col] = split_rows[gene_col].str.split("_")
    split_rows = split_rows.explode(gene_col)
    
    # Combine
    df_expanded = pd.concat([rows_to_keep, split_rows], ignore_index=True)
    
    return df_expanded


###########################################
# Variant annotation functions start here
###########################################

def process_vep_vcf(vcf_file, output_dir, output_filename):
    """
    Processes a VEP-annotated VCF file to filter variants, extract genotypes, 
    and generate structured summary tables including SpliceAI/CADD/ClinVar annotations.

    Parameters:
    - vcf_file (str): Path to the input VCF file.
    - output_dir (str): Directory where output tsv files will be saved.
    - output_filename (str): Base name for the output files (without extension).

    Returns:
    - df (pd.DataFrame): DataFrame containing all filtered variants.
    - df_final (pd.DataFrame): DataFrame summarizing variants grouped by gene and haplotype.
    """

    os.makedirs(output_dir, exist_ok=True)
    vcf = pysam.VariantFile(vcf_file)
    csq_fields = vcf.header.info['CSQ'].description.split(": ")[1].split("|")

    variant_data = []

    for record in vcf:
        chrom = record.chrom
        pos = record.pos
        ref = record.ref
        alts = record.alts
        qual = record.qual

        if qual is None or qual < 30:
            continue

        variant_info = {key: record.info.get(key, None) for key in vcf.header.info.keys() if key != "CSQ"}

        if "CSQ" in record.info:
            for csq_entry in record.info["CSQ"]:
                csq_values = csq_entry.split("|")
                csq_dict = dict(zip(csq_fields, csq_values))

                max_af_str = csq_dict.get("MAX_AF", "")

                if max_af_str in ("", ".", None):
                    max_af = np.nan
                else:
                    try:
                        max_af = float(max_af_str)
                    except (ValueError, TypeError):
                        max_af = np.nan

                # Only filter when an AF is actually known and >= 1%
                if pd.notna(max_af) and max_af >= 0.01:
                    continue

                spliceai_columns = [
                    "SpliceAI_pred_DS_AG", "SpliceAI_pred_DS_AL", 
                    "SpliceAI_pred_DS_DG", "SpliceAI_pred_DS_DL"
                ]
                spliceai_scores = []
                for col in spliceai_columns:
                    value = csq_dict.get(col, "0")
                    try:
                        spliceai_scores.append(float(value))
                    except ValueError:
                        spliceai_scores.append(0)
                highest_spliceai_score = max(spliceai_scores)

                if highest_spliceai_score > 0.5:
                    spliceai_high_score_flag = "SpliceAI_High_Score"
                elif highest_spliceai_score > 0.2:
                    spliceai_high_score_flag = "SpliceAI_Moderate_Score"
                else:
                    spliceai_high_score_flag = ""

                spliceai_pred_symbol = csq_dict.get("SpliceAI_pred_SYMBOL", "")

                # CADD score flag
                cadd_phred_str = csq_dict.get("CADD_PHRED", "0")
                try:
                    cadd_score = float(cadd_phred_str)
                except ValueError:
                    cadd_score = 0.0
                cadd_high_score_flag = "CADD_High_Score" if cadd_score > 15 else ""

                clinvar_clnsig = csq_dict.get("ClinVar_CLNSIG", "")
                clinvar_clndn = csq_dict.get("ClinVar_CLNDN", "")

                # Construct CHROM_POS_REF/ALT
                chrom_pos_refalt = f"{chrom}_{pos}_{ref}/{alts[0]}"

                variant_entry = {
                    "#CHROM_POS_REF/ALT": chrom_pos_refalt,
                    "CHROM": chrom,
                    "POS": pos,
                    "REF": ref,
                    "ALT": alts[0],
                    "QUAL": qual,
                    "MAX_AF": max_af,
                    "Highest_SpliceAI_Score": highest_spliceai_score,
                    "SpliceAI_High_Score_Flag": spliceai_high_score_flag,
                    "SpliceAI_pred_SYMBOL": spliceai_pred_symbol,
                    "CADD_PHRED": cadd_phred_str,
                    "CADD_High_Score_Flag": cadd_high_score_flag,
                    "ClinVar_CLNSIG": clinvar_clnsig,
                    "ClinVar_CLNDN": clinvar_clndn,
                    **variant_info,
                    **csq_dict,
                }

                for sample in vcf.header.samples:
                    genotype_tuple = record.samples[sample]["GT"]
                    is_phased = record.samples[sample].phased

                    if genotype_tuple is not None:
                        separator = "|" if is_phased else "/"
                        genotype_str = separator.join(map(str, genotype_tuple))
                    else:
                        genotype_str = "NA"

                    variant_entry[f"GT_{sample}"] = genotype_str

                variant_data.append(variant_entry)

    df = pd.DataFrame(variant_data)
    df_tsv_path = os.path.join(output_dir, f"{output_filename}_vcf_vep_filtered.tsv")
    df.to_csv(df_tsv_path, index=False, sep="\t")
    print(f"Saved: {df_tsv_path}")

    genotype_columns = [col for col in df.columns if col.startswith("GT_")]
    haplotype_data = {}

    for _, row in df.iterrows():
        gene = row.get("SYMBOL", "")
        variant_id = row["#CHROM_POS_REF/ALT"]
        max_af = row["MAX_AF"]
        clin_sig = row.get("ClinVar_CLNSIG", "")
        clin_disease = row.get("ClinVar_CLNDN", "")
        consequence = row.get("Consequence", "")
        highest_spliceai_score = row["Highest_SpliceAI_Score"]
        spliceai_high_score_flag = row["SpliceAI_High_Score_Flag"]
        spliceai_pred_symbol = row["SpliceAI_pred_SYMBOL"]
        cadd_score = row["CADD_PHRED"]
        cadd_high_score_flag = row["CADD_High_Score_Flag"]

        variant_info = (
            f"({variant_id},{max_af},{clin_sig},{clin_disease},{consequence},"
            f"{highest_spliceai_score},{spliceai_high_score_flag},"
            f"{spliceai_pred_symbol},{cadd_score},{cadd_high_score_flag})"
        )

        if gene not in haplotype_data:
            haplotype_data[gene] = {"Hap1": [], "Hap2": [], "Hap0": []}

        for sample in genotype_columns:
            genotype = row[sample]

            if genotype == "1|0":
                haplotype_data[gene]["Hap1"].append(variant_info)
            elif genotype == "0|1":
                haplotype_data[gene]["Hap2"].append(variant_info)
            elif genotype == "0/1":
                haplotype_data[gene]["Hap0"].append(variant_info)
            elif genotype == "1/1":
                haplotype_data[gene]["Hap1"].append(variant_info)
                haplotype_data[gene]["Hap2"].append(variant_info)

    final_data = [
        {
            "Gene": gene,
            "Hap1": "; ".join(set(data["Hap1"])),
            "Hap2": "; ".join(set(data["Hap2"])),
            "Hap0": "; ".join(set(data["Hap0"]))
        }
        for gene, data in haplotype_data.items()
    ]

    df_final = pd.DataFrame(final_data)
    df_final_tsv_path = os.path.join(output_dir, f"{output_filename}_gene_haplotype_split.tsv")
    df_final.to_csv(df_final_tsv_path, index=False, sep="\t")
    print(f"Saved: {df_final_tsv_path}")

    return df, df_final


def write_sample_gene_lists(
    gene_file,
    isoform_file,
    output_dir="gene_lists_by_sample"
):
    """
    Reads gene and isoform result TSV files, extracts associated genes by sample,
    and writes one gene list per sample into the specified output directory.

    Parameters:
    - gene_file (str): Path to the gene-level TSV file.
    - isoform_file (str): Path to the isoform-level TSV file.
    - output_dir (str): Directory to write per-sample gene list files. Default is "gene_lists_by_sample".
    """
    
    os.makedirs(output_dir, exist_ok=True)

    # Load and concatenate the relevant data
    gene_df = pd.read_csv(gene_file, sep="\t", usecols=["Sample", "associated_gene"])
    iso_df = pd.read_csv(isoform_file, sep="\t", usecols=["Sample", "associated_gene"])
    combined_df = pd.concat([gene_df, iso_df], ignore_index=True)

    # Drop rows with missing data
    combined_df.dropna(subset=["Sample", "associated_gene"], inplace=True)

    # Group by Sample and get unique associated genes
    grouped = combined_df.groupby("Sample")["associated_gene"].apply(lambda x: sorted(set(x)))

    # Write each sample's gene list to a file
    for sample, gene_list in grouped.items():
        safe_sample = sample.replace("/", "_").replace(" ", "_")
        output_path = Path(output_dir) / f"{safe_sample}_gene_list.txt"
        with open(output_path, "w") as f:
            for gene in gene_list:
                f.write(f"{gene}\n")

    print(f"Gene lists written to: {output_dir}")


def passes_max_af_filter(record, max_af_index, max_af_cutoff):
    """
    Checks whether a VCF record passes a maximum allele frequency (MAX_AF) filter
    using annotations from the VEP CSQ field.

    Parameters:
    - record (pysam.VariantRecord): A single variant record from a VCF file annotated with VEP.
    - max_af_index (int): Index of the MAX_AF field in the VEP CSQ annotation string.
    - max_af_cutoff (float): Maximum allowed allele frequency for a variant to be retained.

    Returns:
    - bool: True if any annotation for the variant has a MAX_AF less than or equal to the cutoff; False otherwise.
    """
    if "CSQ" not in record.info:
        return False
    for annotation in record.info["CSQ"]:
        fields = annotation.split("|")
        af_str = fields[max_af_index]
        # Missing / unknown AF -> keep
        if af_str in ("", "."):
            return True
        try:
            if af_str and float(af_str) <= max_af_cutoff:
                return True
        except ValueError:
            continue
    return False

# def filter_multiple_vcfs(pair_list_path, gene_list_dir, flank=1000, max_af_cutoff=0.01,
#                          gtf_path="/mmfs1/gscratch/stergachislab/asedeno/data/reference_files/gencode.v47.annotation.gtf",
#                          output_dir="subsetted_vcfs"):
#     """
#     Filters multiple VEP-annotated VCF files based on gene regions and allele frequency thresholds.
#     Outputs a subsetted and compressed VCF file for each individual.

#     For each row in the pair list TSV, the function:
#     - Locates the individual's VCF file and gene list
#     - Uses a GTF annotation file to find coordinates for genes (with optional flanking region)
#     - Filters variants in the VCF to retain only those within gene regions and passing the MAX_AF threshold
#     - Writes the filtered variants to a compressed VCF file in the output directory

#     Parameters:
#     - pair_list_path (str): Path to a TSV file with at least two columns: 'individual' and 'individual_vcf'.
#     - gene_list_dir (str): Directory containing gene list files named as <individual>_gene_list.txt.
#     - flank (int): Number of base pairs to add upstream and downstream of gene coordinates (default: 1000).
#     - max_af_cutoff (float): Maximum allele frequency for a variant to pass filtering (default: 0.01).
#     - gtf_path (str): Path to a GTF file used to extract gene coordinates.
#     - output_dir (str): Directory where the filtered VCF files will be written (default: "subsetted_vcfs").

#     Returns:
#     - None: Writes filtered VCF files and tabix indexes to the output directory.
#     """
    
#     if not os.path.isfile(pair_list_path):
#         raise FileNotFoundError(f"Pair list file not found: {pair_list_path}")
#     if not os.path.isfile(gtf_path):
#         raise FileNotFoundError(f"GTF file not found: {gtf_path}")
#     if not os.path.isdir(gene_list_dir):
#         raise NotADirectoryError(f"Gene list directory not found: {gene_list_dir}")

#     os.makedirs(output_dir, exist_ok=True)

#     # Parse GTF into gene -> (chrom, start, end)
#     gene_coords = {}
#     with open(gtf_path, 'r') as gtf:
#         for line in gtf:
#             if line.startswith('#'):
#                 continue
#             fields = line.strip().split('\t')
#             if len(fields) < 9 or fields[2] != 'gene':
#                 continue
#             chrom, start, end, attr = fields[0], int(fields[3]), int(fields[4]), fields[8]
#             if 'gene_name' in attr:
#                 gene_name = attr.split('gene_name "')[1].split('"')[0]
#                 if gene_name not in gene_coords:
#                     gene_coords[gene_name] = (chrom, start, end)

#     df = pd.read_csv(pair_list_path, sep="\t")
#     df = df.drop_duplicates(subset=["individual", "individual_vcf"])

#     for idx, row in df.iterrows():
#         individual = row["individual"]
#         gene_list_file = os.path.join(gene_list_dir, f"{individual}_gene_list.txt")
#         vcf_in = row["individual_vcf"]
#         vcf_out = os.path.join(output_dir, f"{individual}.isoranker_subsetted.vcf.gz")

#         if not os.path.isfile(gene_list_file):
#             print(f"  Error: Gene list file not found for individual {individual}: {gene_list_file}. Skipping.")
#             continue
#         if not os.path.isfile(vcf_in):
#             print(f"  Error: VCF file not found: {vcf_in}. Skipping.")
#             continue

#         print(f"Processing individual: {individual}\n  Gene list: {gene_list_file}\n  Input VCF: {vcf_in}\n  Output VCF: {vcf_out}\n  Flank: {flank} bp")

#         with open(gene_list_file) as gf:
#             genes = [g.strip() for g in gf if g.strip()]

#         regions = []
#         for gene in genes:
#             if gene in gene_coords:
#                 chrom, start, end = gene_coords[gene]
#                 regions.append((chrom, max(0, start - flank), end + flank))
#             else:
#                 print(f"  Warning: {gene} not found in GTF.")

#         if not regions:
#             print(f"  No valid regions for {vcf_out}. Skipping.")
#             continue

#         vcf_infile = pysam.VariantFile(vcf_in, "r")  # handles .vcf or .vcf.gz
#         vcf_outfile = pysam.VariantFile(vcf_out, "wz", header=vcf_infile.header)  # write compressed

#         # Get CSQ fields
#         csq_desc = vcf_infile.header.info["CSQ"].description
#         csq_format_str = csq_desc.split("Format: ")[1]
#         csq_fields = csq_format_str.strip().split("|")
#         try:
#             max_af_index = csq_fields.index("MAX_AF")
#         except ValueError:
#             raise ValueError("MAX_AF not found in CSQ field format.")

#         filtered_records = []
#         for chrom, start, end in regions:
#             try:
#                 for record in vcf_infile.fetch(chrom, start, end):
#                     if passes_max_af_filter(record, max_af_index=max_af_index, max_af_cutoff=max_af_cutoff):
#                         filtered_records.append(record)
#             except ValueError as e:
#                 print(f"  Warning: Region {chrom}:{start}-{end} not found in VCF: {e}")

#         filtered_records.sort(key=lambda r: (r.contig, r.pos))
#         for record in filtered_records:
#             vcf_outfile.write(record)

#         vcf_outfile.close()

#         try:
#             pysam.tabix_index(vcf_out, preset="vcf", force=True)
#             print(f"Indexed: {vcf_out}.tbi")
#         except Exception as e:
#             print(f"  Error indexing {vcf_out}: {e}")

#         print(f"Finished: {vcf_out}\n-----------------------------")




def filter_multiple_vcfs(pair_list_path, gene_list_dir, flank=1000, max_af_cutoff=0.01,
                         gtf_path="/mmfs1/gscratch/stergachislab/asedeno/data/reference_files/gencode.v47.annotation.gtf",
                         output_dir="subsetted_vcfs"):

    import os
    import gzip
    import shutil
    import tempfile
    import pandas as pd
    import pysam

    ################################################
    # Helper
    ################################################

    def is_gzip_file(path):
        """
        Detect whether a file is truly gzip-compressed
        using magic bytes rather than filename suffix.
        """
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"

    ################################################
    # Validate inputs
    ################################################

    if not os.path.isfile(pair_list_path):
        raise FileNotFoundError(f"Pair list file not found: {pair_list_path}")

    if not os.path.isfile(gtf_path):
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")

    if not os.path.isdir(gene_list_dir):
        raise NotADirectoryError(f"Gene list directory not found: {gene_list_dir}")

    os.makedirs(output_dir, exist_ok=True)

    ################################################
    # Parse GTF into gene coordinates
    ################################################

    gene_coords = {}

    with open(gtf_path, "r") as gtf:

        for line in gtf:

            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")

            if len(fields) < 9:
                continue

            if fields[2] != "gene":
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            attr = fields[8]

            if 'gene_name "' not in attr:
                continue

            gene_name = attr.split('gene_name "')[1].split('"')[0]

            if gene_name not in gene_coords:
                gene_coords[gene_name] = (chrom, start, end)

    ################################################
    # Read pair list
    ################################################

    df = pd.read_csv(pair_list_path, sep="\t")

    df = df.drop_duplicates(
        subset=["individual", "individual_vcf"]
    )

    ################################################
    # Process each sample
    ################################################

    for _, row in df.iterrows():

        individual = row["individual"]
        vcf_in = row["individual_vcf"]

        if pd.isna(vcf_in) or str(vcf_in).strip() == "" or str(vcf_in).strip().lower() in {"nan", "none", "na"}:
            print(
                f"Skipping {individual}: no individual_vcf provided.",
                flush=True
            )
            continue

        vcf_in = str(vcf_in).strip()

        gene_list_file = os.path.join(
            gene_list_dir,
            f"{individual}_gene_list.txt"
        )

        vcf_out = os.path.join(
            output_dir,
            f"{individual}.isoranker_subsetted.vcf.gz"
        )


        vcf_index = vcf_out + ".tbi"

        if (
            os.path.exists(vcf_out)
            and os.path.getsize(vcf_out) > 0
            and os.path.exists(vcf_index)
        ):
            print(
                f"Skipping {individual}: subsetted VCF already exists",
                flush=True
            )
            continue

        ################################################
        # Validate files
        ################################################

        if not os.path.isfile(gene_list_file):

            print(
                f"Error: Gene list file not found for {individual}: {gene_list_file}",
                flush=True
            )

            continue

        if not os.path.isfile(vcf_in):

            print(
                f"Error: VCF file not found: {vcf_in}",
                flush=True
            )

            continue

        print(
            f"Processing individual: {individual}",
            flush=True
        )

        print(
            f"Input VCF: {vcf_in}",
            flush=True
        )

        ################################################
        # Read genes
        ################################################

        with open(gene_list_file) as gf:

            genes = [
                g.strip()
                for g in gf
                if g.strip()
            ]

        ################################################
        # Build genomic regions
        ################################################

        regions = []

        for gene in genes:

            if gene in gene_coords:

                chrom, start, end = gene_coords[gene]

                regions.append((
                    chrom,
                    max(0, start - flank),
                    end + flank
                ))

            else:

                print(
                    f"Warning: {gene} not found in GTF.",
                    flush=True
                )

        if not regions:

            print(
                f"No valid regions for {individual}. Skipping.",
                flush=True
            )

            continue

        ################################################
        # Initialize cleanup variables
        ################################################

        temp_dir = None
        vcf_infile = None
        vcf_outfile = None

        try:

            ################################################
            # Check whether original VCF is usable directly
            ################################################

            has_index = (
                os.path.exists(vcf_in + ".tbi")
                or os.path.exists(vcf_in + ".csi")
            )

            use_original = False

            if has_index:

                try:

                    test_vcf = pysam.VariantFile(vcf_in, "r")

                    chrom, start, end = regions[0]

                    list(
                        test_vcf.fetch(
                            chrom,
                            start,
                            end
                        )
                    )

                    test_vcf.close()

                    use_original = True

                    print(
                        "Using original indexed VCF.",
                        flush=True
                    )

                except Exception as e:

                    print(
                        f"Original VCF/index unusable: {e}",
                        flush=True
                    )

                    use_original = False

            ################################################
            # Create temporary bgzip VCF if needed
            ################################################

            if use_original:

                indexed_vcf = vcf_in

            else:

                print(
                    "Creating temporary bgzip-compressed indexed VCF.",
                    flush=True
                )

                base_tmp = os.environ.get(
                    "TMPDIR",
                    tempfile.gettempdir()
                )

                temp_dir = tempfile.mkdtemp(
                    prefix="isoranker_",
                    dir=base_tmp
                )

                local_vcf_bgz = os.path.join(
                    temp_dir,
                    os.path.basename(vcf_in)
                )

                if not local_vcf_bgz.endswith(".gz"):
                    local_vcf_bgz += ".gz"

                ################################################
                # Rewrite as proper BGZF
                ################################################

                with pysam.BGZFile(local_vcf_bgz, "wb") as out_f:

                    if is_gzip_file(vcf_in):

                        print(
                            "Detected true gzip-compressed VCF.",
                            flush=True
                        )

                        with gzip.open(vcf_in, "rb") as in_f:
                            shutil.copyfileobj(in_f, out_f)

                    else:

                        print(
                            "Detected plain-text VCF with misleading extension.",
                            flush=True
                        )

                        with open(vcf_in, "rb") as in_f:
                            shutil.copyfileobj(in_f, out_f)

                ################################################
                # Create tabix index
                ################################################

                pysam.tabix_index(
                    local_vcf_bgz,
                    preset="vcf",
                    force=True,
                    keep_original=True
                )

                indexed_vcf = local_vcf_bgz

                print(
                    f"Temporary BGZF VCF: {indexed_vcf}",
                    flush=True
                )

                print(
                    f"Temporary index: {indexed_vcf}.tbi",
                    flush=True
                )

            ################################################
            # Open indexed VCF
            ################################################

            vcf_infile = pysam.VariantFile(
                indexed_vcf,
                "r"
            )

            vcf_outfile = pysam.VariantFile(
                vcf_out,
                "wz",
                header=vcf_infile.header
            )

            ################################################
            # Parse CSQ fields
            ################################################

            csq_desc = vcf_infile.header.info["CSQ"].description

            if "Format: " in csq_desc:
                csq_format_str = csq_desc.split("Format: ")[1]
            else:
                csq_format_str = csq_desc.split(": ")[1]

            csq_fields = csq_format_str.strip().split("|")

            try:

                max_af_index = csq_fields.index("MAX_AF")

            except ValueError:

                raise ValueError(
                    "MAX_AF not found in CSQ field format."
                )

            ################################################
            # Fetch records
            ################################################

            filtered_records = []

            for chrom, start, end in regions:

                try:

                    for record in vcf_infile.fetch(
                        chrom,
                        start,
                        end
                    ):

                        if passes_max_af_filter(
                            record,
                            max_af_index=max_af_index,
                            max_af_cutoff=max_af_cutoff
                        ):

                            filtered_records.append(record)

                except Exception as e:

                    print(
                        f"Warning: Region {chrom}:{start}-{end} failed: {e}",
                        flush=True
                    )

            ################################################
            # Sort records
            ################################################

            filtered_records.sort(
                key=lambda r: (r.contig, r.pos)
            )

            ################################################
            # Write records
            ################################################

            for record in filtered_records:
                vcf_outfile.write(record)

            vcf_outfile.close()
            vcf_outfile = None

            vcf_infile.close()
            vcf_infile = None

            ################################################
            # Index output VCF
            ################################################

            pysam.tabix_index(
                vcf_out,
                preset="vcf",
                force=True,
                keep_original=True
            )

            print(
                f"Finished: {vcf_out}",
                flush=True
            )

            print(
                f"Indexed output: {vcf_out}.tbi",
                flush=True
            )

            print(
                "-----------------------------",
                flush=True
            )

        ################################################
        # Cleanup
        ################################################

        finally:

            if vcf_outfile is not None:
                vcf_outfile.close()

            if vcf_infile is not None:
                vcf_infile.close()

            if temp_dir is not None:

                if os.path.isdir(temp_dir):

                    shutil.rmtree(temp_dir)

                    print(
                        f"Removed temp directory: {temp_dir}",
                        flush=True
                    )