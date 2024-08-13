# The idea of this script is to test how much money we would have made or lost
# by using the 2017-2018 season and betting to make £2 whenever we find value

from collections import namedtuple
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import fifa_ratings_predictor.constants as constants
from fifa_ratings_predictor.data_methods import read_match_data, read_player_data, normalise_features, \
    assign_odds_to_match, read_all_football_data, get_match_odds, get_match_odds_max
from fifa_ratings_predictor.matching import match_lineups_to_fifa_players, create_feature_vector_from_players
from fifa_ratings_predictor.model import NeuralNet, goal_diff_to_outcome, RandomPredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

Bet = namedtuple('Bet', ['true_odds', 'predicted_odds', 'stake', 'type', 'profit', 'match'])


class BetTracker:
    def __init__(self):
        self.invested = 0
        self.pending_bet = None
        self.completed_bets = []
        self.profit = 0
        self.bankroll = 100
        self.pocket = 0

        self.completed_odds = []

        self.bankroll_threshold = 120
        self.withdraw = 0

    def make_bet(self, bet):
        self.pending_bet = bet
        self.invested += bet.stake
        self.bankroll -= bet.stake

    def bet_won(self):
        self.completed_odds.append(self.pending_bet.true_odds - 1)
        self.profit += self.pending_bet.profit
        self.bankroll += self.pending_bet.stake + self.pending_bet.profit
        self.completed_bets.append((self.pending_bet, 'W'))
        self.pending_bet = None

    def update_pocket(self):
        if self.withdraw <= 0:
            return
        while self.bankroll > self.bankroll_threshold:
            self.pocket += self.withdraw
            self.bankroll -= self.withdraw


    def bet_lost(self):
        self.completed_odds.append(-1)
        self.profit -= self.pending_bet.stake
        self.completed_bets.append((self.pending_bet, 'L'))
        self.pending_bet = None

    @property
    def roi(self):
        return self.profit / self.invested


def calculate_profit(bet):
    return bet.stake * bet.odds - bet.stake


def calculate_stake(odds, method='constant profit', constant_profit=2, probability=None):
    assert method in ['constant_profit', 'kelly']
    if method == 'constant_profit':
        stake = constant_profit / (odds - 1)
    elif method == 'kelly':
        stake = ((odds * probability) - 1) / (odds - 1)
    return stake



def calc_match_data_and_features(league='F1',league_model='F1',season='2013-2014', season_model='2016-2017'):
    match_data = read_match_data(season=season, league=league)

    match_data = assign_odds_to_match(match_data, read_all_football_data(league=league))

    player_data = read_player_data(season=season)

    errors = []

    cached_players = {}

    feature_vectors = []

    for match in match_data:

        try:

            home_players_matched, cached_players = match_lineups_to_fifa_players(match['info']['home lineup names'],
                                                                                 match['info']['home lineup raw names'],
                                                                                 match['info']['home lineup numbers'],
                                                                                 match['info'][
                                                                                     'home lineup nationalities'],
                                                                                 constants.LINEUP_TO_PLAYER_TEAM_MAPPINGS[
                                                                                     'ALL'][
                                                                                     match['info']['home team']],
                                                                                 match['info']['season'],
                                                                                 player_data, cached_players)

            away_players_matched, cached_players = match_lineups_to_fifa_players(match['info']['away lineup names'],
                                                                                 match['info']['away lineup raw names'],
                                                                                 match['info']['away lineup numbers'],
                                                                                 match['info'][
                                                                                     'away lineup nationalities'],
                                                                                 constants.LINEUP_TO_PLAYER_TEAM_MAPPINGS[
                                                                                     'ALL'][
                                                                                     match['info']['away team']],
                                                                                 match['info']['season'],
                                                                                 player_data, cached_players)

            home_feature_vector = create_feature_vector_from_players(home_players_matched)
            away_feature_vector = create_feature_vector_from_players(away_players_matched)
            #
            home_odds, draw_odds, away_odds  = get_match_odds(match)
            home_odds_max, draw_odds_max, away_odds_max  = get_match_odds_max(match)
            #
            feature_vector = home_feature_vector + away_feature_vector
            feature_vector.extend([1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds])
            #
            feature_vector = np.array(feature_vector).reshape(-1, len(feature_vector))

            feature_vectors.append(normalise_features(feature_vector))

        except Exception as exception:
            print(match['info']['date'], match['info']['home team'], match['info']['away team'])
            print(exception)
            errors.append(match['match number'])

    feature_vectors = np.vstack((x for x in feature_vectors))

    match_data = [match for match in match_data if match['match number'] not in errors]
    return match_data, feature_vectors


def main(league='F1',league_model='F1',season='2013-2014', season_model='2016-2017'):
    #
    probs0 = np.load(f'./data/lineup-data/{league}/processed-numpy-arrays/probs-{season}.npy')

    # net = NeuralNet()
    net = RandomPredictor()

    bank = [100]

    all_odds = []

    match_data_dump = Path(f"backtest-match_data-{league}-{season}.v1.pickle")
    features_dump = Path(f"backtest-features_vector-{league}-{season}.v1.pickle")
    if match_data_dump.exists() and features_dump.exists():
        match_data = pickle.loads(match_data_dump.read_bytes())
        feature_vectors = pickle.loads(features_dump.read_bytes())
    else:
        match_data, feature_vectors = calc_match_data_and_features(league, league_model, season, season_model)
        match_data_dump.write_bytes(pickle.dumps(match_data))
        features_dump.write_bytes(pickle.dumps(feature_vectors))
    #
    bet_tracker = BetTracker()

    #probabilities = net.predict(feature_vectors, model_name=f'./models/{league_model}-{season_model}' + '/deep')


    goal_diffs = net.predict(feature_vectors, model_name=f'./models/{league_model}-{season_model}' + '/deep')

    # for match, probability in zip(match_data, probabilities):
    for match, goal_diff in zip(match_data, goal_diffs):
        outcome = goal_diff_to_outcome(goal_diff.reshape(-1, 1))

        # import pdb;
        # pdb.set_trace()

        #
        probability = probs0 * outcome
        probability *= 0.85
        probability = probability.flatten()

        # print(match['info']['date'], match['info']['home team'], match['info']['away team'])

        pred_home_odds, pred_draw_odds, pred_away_odds = [1/x for x in probability]

        home_odds, draw_odds, away_odds = match['info']['home odds'], match['info']['draw odds'], match['info'][
            'away odds']
        if any(np.isnan([home_odds, draw_odds, away_odds])):
            # print(home_odds, draw_odds, away_odds)
            continue

        home_odds_max, draw_odds_max, away_odds_max = match['info']['home odds max'], match['info']['draw odds max'], match['info'][
            'away odds max']

        all_odds.append((pred_home_odds, home_odds))
        all_odds.append((pred_away_odds, away_odds))
        #
        imax = probability.argmax()

        #if (probability[0] > 0.51) | (pred_home_odds < home_odds*0.97 < 3.2):
        if (probability[0] > 0.1):
        #if (pred_home_odds < home_odds < 3.2) & (0.02 <= probability[0] - 1 / home_odds):
        #if imax == 0:
            #import pdb; pdb.set_trace()
            stake = calculate_stake(home_odds, probability=probability[0], method='kelly',
                                    constant_profit=20) * bet_tracker.bankroll
            stake = 1
            if (stake < 0.001*bet_tracker.bankroll):
                continue
            #
            #if (pred_home_odds < home_odds*0.9 < 3.2):
            #    stack = max(0.1*bet_tracker.bankroll,stake)

            #profit = stake * home_odds - stake
            profit = stake * home_odds_max - stake
            bet = Bet(true_odds=home_odds, predicted_odds=pred_home_odds, stake=stake, profit=profit, match=match,
                      type='home')
            bet_tracker.make_bet(bet)
            # #
            # print(probability)
            # print(f'stake:{stake},profit:{profit}')
            # print(stake * home_odds_max - stake,stake * home_odds - stake)
            # print(home_odds, draw_odds, away_odds)
            # print(match['info']['home goals'],match['info']['away goals'])
            # print(stake,profit,bet_tracker.bankroll,bet_tracker.invested)
            #
            if match['info']['home goals'] > match['info']['away goals']:
                bet_tracker.bet_won()
            else:
                bet_tracker.bet_lost()
            bank.append(bet_tracker.bankroll + bet_tracker.pocket)
        #if (probability[2] > 0.51) | (pred_away_odds < away_odds*0.97 < 3.2):
        if (probability[2] > 0.1):
        #elif (pred_away_odds < away_odds < 3.2) & (0.02 <= probability[2] - 1 / away_odds):
        #if imax == 1:
            #import pdb; pdb.set_trace()
            stake = calculate_stake(away_odds, probability=probability[2], method='kelly',
                                    constant_profit=20) * bet_tracker.bankroll
            stake = 1
            if (stake < 0.001*bet_tracker.bankroll):
                continue
            #
            #if (pred_away_odds < away_odds*0.9 < 3.2):
            #    stack = max(0.1*bet_tracker.bankroll,stake)

            #profit = stake * away_odds - stake
            profit = stake * away_odds_max - stake
            bet = Bet(true_odds=away_odds, predicted_odds=pred_away_odds, stake=stake, profit=profit, match=match,
                      type='away')
            bet_tracker.make_bet(bet)
            #
            # print(probability)
            # print(f'stake:{stake},profit:{profit}')
            # print(stake * home_odds_max - stake,stake * home_odds - stake)
            # print(home_odds, draw_odds, away_odds)
            # print(match['info']['home goals'],match['info']['away goals'])
            # print(stake,profit,bet_tracker.bankroll,bet_tracker.invested)
            #
            if match['info']['home goals'] < match['info']['away goals']:
                bet_tracker.bet_won()
            else:
                bet_tracker.bet_lost()
            bank.append(bet_tracker.bankroll + bet_tracker.pocket)
        #if (probability[1] > 0.51) | (pred_draw_odds < draw_odds*0.97 < 3.2):
        if (probability[1] > 0.1):
        #elif (pred_draw_odds < draw_odds < 3.2) & (0.02 <= probability[1] - 1 / draw_odds):
        #if imax == 2:
            #import pdb; pdb.set_trace()
            stake = calculate_stake(draw_odds, probability=probability[1], method='kelly',
                                    constant_profit=20) * bet_tracker.bankroll
            stake = 1
            if (stake < 0.001*bet_tracker.bankroll):
                continue
            #
           #if (pred_draw_odds < draw_odds*0.9 < 3.2):
           #    stack = max(0.1*bet_tracker.bankroll,stake)
            #
            profit = stake * draw_odds_max - stake
            #profit = stake * draw_odds - stake
            bet = Bet(true_odds=draw_odds, predicted_odds=pred_draw_odds, stake=stake, profit=profit, match=match,
                      type='draw')
            bet_tracker.make_bet(bet)
            #
            # print(probability)
            # print(f'stake:{stake},profit:{profit}')
            # print(stake * home_odds_max - stake,stake * home_odds - stake)
            # print(home_odds, draw_odds, away_odds)
            # print(match['info']['home goals'],match['info']['away goals'])
            # print(stake,profit,bet_tracker.bankroll,bet_tracker.invested)
            #
            if match['info']['home goals'] == match['info']['away goals']:
                bet_tracker.bet_won()
            else:
                bet_tracker.bet_lost()
            bank.append(bet_tracker.bankroll + bet_tracker.pocket)

        bet_tracker.update_pocket()
        # print("bankroll:", bet_tracker.bankroll, bet_tracker.completed_bets[-1])
        # if np.isnan(bet_tracker.bankroll):
        #     break

    return bet_tracker, bank, all_odds


def plot_backtest(bankroll, roi, plot_title, name='graph.png'):
    import matplotlib.font_manager
    import matplotlib as mpl

    font = {'size': 6}

    mpl.rc('font', **font)
    mpl.rcParams.update({'text.color': "#333344",
                         'axes.labelcolor': "#333344"})

    flist = matplotlib.font_manager.get_fontconfig_fonts()
    for fname in flist:
        try:
            s = matplotlib.font_manager.FontProperties(fname=fname).get_name()
            #if 'bank' in s:
            #    props = matplotlib.font_manager.FontProperties(fname=fname)
        except RuntimeError:
            pass

    propies = dict(boxstyle='round', facecolor='#225599')
    fig = plt.figure()
    ax = plt.axes()
    plt.plot(np.arange(len(bankroll)), bankroll, c='#113355')
    ax.text(0.05, 0.95, 'ROI: {0:.2%}'.format(roi), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=propies, color='white')
    fig.set_facecolor('#aabbcc')
    ax.set_facecolor('#aabbcc')
    ax.set_title(plot_title, color="#223355")
    plt.savefig(name, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())


if __name__ == '__main__':
    leagues=['SP1','D1','F1','E0']
    leagues = ['F1']
    league_model='F1'
    #season_model = '2013-2014'
    #seasons = ['2013-2014', '2014-2015', '2015-2016', '2016-2017', '2017-2018']
    seasons = ['2016-2017', '2017-2018']
    ##
    for league in leagues:
        for season in seasons:
            print(league,league,season)
            #print(league,league_model,season)
            #tracker, bankroll, odds = main(league,league_model,season, season_model)
            tracker, bankroll, odds = main(league, league, season, season)
            print(league, season, np.mean(tracker.completed_odds), np.std(tracker.completed_odds))
            print(league, season, "sum", np.sum(tracker.completed_odds))
            # print(tracker.completed_odds)
            # import pdb;
            # pdb.set_trace()
            plot_backtest(bankroll, tracker.roi, f'graph_{league}_{season}', f'graph_{league}_{season}.png')
