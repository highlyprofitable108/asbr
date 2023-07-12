import os
import pandas as pd
from constants import constants
from utils.file_utils import get_most_recent_file, get_timestamp


def load_data(directory):
    csv_file = get_most_recent_file(directory, "golf*.csv")
    if csv_file is None:
        print("No CSV file found.")
        return None
    df = pd.read_csv(csv_file)
    if 'sim' not in df.columns:
        print(f"'sim' column not found in {csv_file}.")
        return None
    return df


def process_data(df, num_simulations):
    # Identify the best player based on round and total score for each sim
    best_player_round_1 = df.loc[df.groupby(
        'sim'
    )['round1'].idxmin()][['dg_id', 'player_name']]
    best_player_total = df.loc[df.groupby(
        'sim'
    )['total_score'].idxmin()][['dg_id', 'player_name']]

    # Count the number of times each player has the lowest score
    best_player_round_1_counts = best_player_round_1.value_counts()
    best_player_total_counts = best_player_total.value_counts()

    # Create DataFrame from the counts
    df_grouped = pd.DataFrame(
        {
            'round1_low_score_count': best_player_round_1_counts,
            'total_low_score_count': best_player_total_counts
        }
    )

    # Adding 'player_name' back to df_grouped
    df_grouped = df_grouped.merge(
        df[['dg_id', 'player_name']].drop_duplicates(), on='dg_id', how='left'
    )

    df_grouped['first_round_vegas'] = num_simulations / df_grouped[
        'round1_low_score_count'
    ]
    df_grouped['tournament_vegas'] = num_simulations / df_grouped[
        'total_low_score_count'
    ]

    # Convert to American odds
    df_grouped.loc[
        df_grouped['first_round_vegas'] >= 2, 'first_round_vegas'
    ] = (df_grouped['first_round_vegas'] - 1) * 100
    df_grouped.loc[
        df_grouped['first_round_vegas'] < 2, 'first_round_vegas'
    ] = -100 / (df_grouped['first_round_vegas'] - 1)

    df_grouped.loc[
        df_grouped['tournament_vegas'] >= 2, 'tournament_vegas'
    ] = (df_grouped['tournament_vegas'] - 1) * 100
    df_grouped.loc[
        df_grouped['tournament_vegas'] < 2, 'tournament_vegas'
    ] = -100 / (df_grouped['tournament_vegas'] - 1)

    df_grouped['first_round_vegas'] = df_grouped[
        'first_round_vegas'
    ].apply(
        lambda odds: f"+{int(odds)}" if not pd.isna(
            odds
        ) and odds > 0 else f"{int(odds)}" if not pd.isna(
            odds
        ) else "+1000000"
        )
    df_grouped['tournament_vegas'] = df_grouped[
        'tournament_vegas'
    ].apply(
        lambda odds: f"+{int(odds)}" if not pd.isna(
            odds
        ) and odds > 0 else f"{int(odds)}" if not pd.isna(
            odds
        ) else "+1000000"
    )

    # Set NaN values in calculated odds columns to +1000000
    df_grouped['calculated_first_odds'] = df_grouped[
        'round1_low_score_count'
    ].apply(
        lambda count: num_simulations / count
    ).fillna(1000000)
    df_grouped['calculated_total_odds'] = df_grouped[
        'total_low_score_count'
    ].apply(
        lambda count: num_simulations / count
    ).fillna(1000000)

    # Reorder the columns and rename the first column
    df_grouped = df_grouped.rename(columns={'dg_id': 'rank'}).reindex(columns=[
        'rank', 'player_name', 'first_round_vegas', 'tournament_vegas',
        'round1_low_score_count', 'total_low_score_count',
        'calculated_first_odds', 'calculated_total_odds'
    ])

    return df_grouped.reset_index(drop=True)


def write_to_csv(df, output_directory, file_name):
    """
    Writes a DataFrame to a csv file in the given directory with a timestamp.

    Parameters:
    df (pd.DataFrame): The DataFrame to write to a csv.
    output_directory (str): The directory to write the csv to.
    file_name (str): The name of the csv file.

    """
    df.to_csv(
        os.path.join(
            output_directory, f'{file_name}_{get_timestamp()}.csv'
        ), index=False
    )


if __name__ == "__main__":
    output_directory = constants['output_directory']

    df = load_data(output_directory)
    if df is not None:
        num_simulations = df['sim'].nunique()
        processed_df = process_data(df, num_simulations)
        write_to_csv(processed_df, output_directory, 'processed')
