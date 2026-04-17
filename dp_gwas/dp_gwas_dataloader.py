import pandas as pd

def load_gwas_data(file_path: str, chr_min=1, chr_max=22, alpha=5e-8):

    dfs = []

    for chr in range(chr_min, chr_max + 1):
        filename = f"{file_path}/ihac2.chr{chr}.eur.meta"
        df = pd.read_csv(filename, sep=r'\s+', engine='python')
        df = df.dropna(subset=['P', 'BP']).query('P > 0')
        df['CHR'] = chr
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.sort_values('BP').reset_index(drop=True)
    df['ind'] = range(len(df))
    df['significant'] = df['P'] < alpha
    
    n_snps = len(df)
    n_causal = df['significant'].sum()
    n_non_causal = n_snps - n_causal
    n_causal_ratio = n_causal / n_snps
    n_non_causal_ratio = n_non_causal / n_snps

    print(f"Number of SNPs: {n_snps}")
    print(f"Number of causal SNPs: {n_causal}")
    print(f"Number of non-causal SNPs: {n_non_causal}")
    print(f"Causal ratio: {n_causal_ratio}")
    print(f"Non-causal ratio: {n_non_causal_ratio}")

    df_causal = df[df['significant']]
    print(df_causal.head())

    return df, n_snps, n_causal, n_non_causal, n_causal_ratio, n_non_causal_ratio

if __name__ == "__main__":
    df, n_snps, n_causal, n_non_causal, n_causal_ratio, n_non_causal_ratio = load_gwas_data(file_path="../datasets/IHAC_meta", chr_min=1, chr_max=4, alpha=5e-8)
    