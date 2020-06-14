import os
import pandas as pd
import swifter

import pendulum as pdl

if __name__ == '__main__':
    if not os.path.exists('result'):
        os.mkdir('result')

    df_tr = pd.read_csv('./data/train_need_aggregate.csv')
    df_te = pd.read_csv('./data/test_need_aggregate.csv')

    df_tr['datetime'] = (
        df_tr['datetime']
        .swifter
        .apply(
            lambda x: pdl.parse(x).format('YYYY-MM-DD HH:mm:00')
        )
    )

    df_te['datetime'] = (
        df_te['datetime']
        .swifter
        .apply(
            lambda x: pdl.parse(x).format('YYYY-MM-DD HH:mm:00')
        )
    )

    df_tr_agg = (
        df_tr
        .groupby('datetime')['EventId']
        .apply(list)
        .reset_index(name='EventId')
        .to_csv('./result/train.csv', index=False)
    )

    df_te_agg = (
        df_te
        .groupby('datetime')['EventId']
        .apply(list)
        .reset_index(name='EventId')
        .to_csv('./result/test.csv', index=False)
    )



