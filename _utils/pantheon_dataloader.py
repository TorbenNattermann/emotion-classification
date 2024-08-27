import pandas as pd


def fetch_pantheon():
    variant = pd.read_parquet('/home/tnattermann/repos/pantheon-kitt-v3/data/s3_data/contextclf202312051230/variant_ceres.parquet')
    referrer = pd.read_parquet('/home/tnattermann/repos/pantheon-kitt-v3/data/s3_data/contextclf202312051230/referrer_ceres.parquet')
    # reconstruct the original text used for embedding inference
    variant['text'] = variant.apply(lambda row: "{}\n{}\n{}".format(row['attr_playout_teaser'],
                                                                    row['attr_playout_title'],
                                                                    row['attr_playout_content']),
                                    axis=1)
    referrer['text'] = referrer['attr_page_content'].str.slice(0, 15000)
    return [variant, referrer]



if __name__ == "__main__":
    data = fetch_pantheon()
    print(data[0].text[:5])
    print(data[1].text[:5])