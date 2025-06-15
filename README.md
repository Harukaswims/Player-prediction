# Player-prediction
#  IPL Best XI Predictor

An advanced AI-powered system for predicting the optimal playing XI for IPL fantasy cricket using machine learning models and comprehensive statistical analysis.

##  Features

- ** AI-Powered Predictions**: Uses ensemble machine learning models (Gradient Boosting + Random Forest)
- ** Comprehensive Analysis**: Analyzes batting, bowling, and fielding performance
- ** Smart Team Selection**: Optimizes team composition with role-based constraints
- ** Budget Management**: Considers player costs and value-for-money metrics
- ** Form Analysis**: Evaluates recent form and consistency patterns
- ** Captain Suggestions**: Recommends captaincy choices based on predicted performance
- ** Detailed Reports**: Generates comprehensive team analysis reports

## Installation

### Prerequisites

```bash
Python 3.8+
```

### Required Libraries

```bash
pip install pandas numpy scikit-learn
```

### Dependencies

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
typing
dataclasses
datetime
logging
warnings
```

## Dataset Requirements

The system requires IPL dataset with the following files:

### 1. matches.csv
Required columns:
- `id` - Match ID
- `season` - IPL season year
- `team1` - First team
- `team2` - Second team
- `winner` - Match winner

### 2. deliveries.csv
Required columns:
- `match_id` - Match ID (linking to matches.csv)
- `inning` - Inning number
- `batting_team` - Batting team name
- `bowling_team` - Bowling team name
- `over` - Over number
- `ball` - Ball number in over
- `batsman` - Batsman name
- `bowler` - Bowler name
- `batsman_runs` - Runs scored by batsman
- `extra_runs` - Extra runs
- `total_runs` - Total runs in delivery

Optional columns (auto-generated if missing):
- `is_wicket` - Wicket indicator
- `dismissal_kind` - Type of dismissal
- `player_dismissed` - Dismissed player

## Quick Start

### Basic Usage

```python
import pandas as pd
from ipl_best_xi_predictor import CricketDataAnalyzer, AIModelPredictor, BestXIPredictor

# Load IPL dataset
matches_df = pd.read_csv('matches.csv')
deliveries_df = pd.read_csv('deliveries.csv')

# Initialize the system
analyzer = CricketDataAnalyzer(matches_df, deliveries_df)
ai_predictor = AIModelPredictor()
predictor = BestXIPredictor(analyzer, ai_predictor)

# Generate Best XI prediction
best_xi = predictor.predict_best_xi()

# Generate detailed report
report = predictor.generate_team_report(best_xi)
print(report)

# Save report to file
with open('best_xi_report.txt', 'w') as f:
    f.write(report)
```

### Advanced Usage

```python
# Filter by specific teams
team_filter = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore']
best_xi = predictor.predict_best_xi(team_filter=team_filter)

# Add match context for better predictions
match_context = {
    'venue_batting_avg': 160,
    'venue_bowling_avg': 28,
    'weather_factor': 1.1,
    'pitch_factor': 0.9
}
best_xi = predictor.predict_best_xi(match_context=match_context)

# Train AI models with custom data
training_data = [
    {
        'stats': player_stats_dict,
        'match_context': context_dict,
        'actual_batting_points': 25,
        'actual_bowling_points': 30,
        'actual_fielding_points': 8
    }
    # ... more training data
]
ai_predictor.train_models(training_data)
```

##  System Architecture

### Core Components

1. **CricketDataAnalyzer**
   - Processes IPL dataset
   - Calculates player statistics
   - Handles data validation and cleaning

2. **AIModelPredictor**
   - Trains machine learning models
   - Predicts player performance
   - Calculates confidence scores

3. **AdvancedFantasyCalculator**
   - Calculates fantasy points
   - Applies scoring rules and bonuses
   - Considers recent form factors

4. **BestXIPredictor**
   - Orchestrates team selection
   - Applies constraints and optimization
   - Generates comprehensive reports

### Player Classification

- **WK (Wicket Keeper)**: 1 required
- **BAT (Batsman)**: 3-6 players
- **AR (All-Rounder)**: 1-4 players
- **BOWL (Bowler)**: 3-6 players

### Team Constraints

- **Total Players**: Exactly 11
- **Maximum per IPL Team**: 7 players
- **Budget Cap**: 100 credits (simulated)
- **Role Balance**: Maintains optimal team composition

## Machine Learning Models

### Batting Prediction
- **Algorithm**: Gradient Boosting Regressor
- **Features**: Historical runs, strike rate, consistency, form
- **Target**: Expected batting fantasy points

### Bowling Prediction
- **Algorithm**: Random Forest Regressor
- **Features**: Wickets, economy rate, strike rate, recent form
- **Target**: Expected bowling fantasy points

### Fielding Prediction
- **Algorithm**: Random Forest Regressor
- **Features**: Catches, run-outs, player activity
- **Target**: Expected fielding fantasy points

## Performance Metrics

The system tracks multiple performance indicators:

- **Predicted Points**: AI-generated performance score
- **Confidence Score**: Model certainty (0.3 - 0.95)
- **Recent Form**: Performance trend (0.3 - 2.0)
- **Value for Money**: Points per cost ratio
- **Consistency Score**: Performance stability metric

## Sample Output

```
================================================================================
 AI-POWERED IPL BEST XI PREDICTION
================================================================================

 TEAM COMPOSITION:
Total Players: 11
Wicket Keepers: 1
Batsmen: 4
All-Rounders: 3
Bowlers: 3

 TEAM METRICS:
Total Predicted Points: 421.7
Average Points per Player: 38.3
Total Cost: 95.2 Cr
Average Cost per Player: 8.7 Cr

 SELECTED PLAYERS:
 1. Virat Kohli        (RCB)
    Role: BAT  | Points:  45.2 | Cost:  9.5 | Form: 1.15
    Batting: 38.5 | Bowling:  2.1 | Fielding:  4.6
    Confidence: 0.87 | Value: 4.76

 KEY INSIGHTS:
• 6 players with high confidence (>0.8)
• 4 players in excellent recent form (>1.2)
• 7 excellent value-for-money picks

 CAPTAINCY SUGGESTIONS:
Captain: Virat Kohli (45.2 points)
Vice-Captain: Jasprit Bumrah (42.8 points)
```

## Configuration

### Scoring Rules

```python
scoring_rules = {
    'batting': {
        'run': 1,
        'boundary': 1,
        'six': 2,
        'fifty': 8,
        'century': 16,
        'duck': -2
    },
    'bowling': {
        'wicket': 12,
        'maiden': 4,
        'five_wickets': 16
    },
    'fielding': {
        'catch': 4,
        'run_out': 6,
        'stumping': 6
    }
}
```

### Team Constraints

```python
team_constraints = {
    'total_players': 11,
    'max_per_team': 7,
    'min_batsmen': 3,
    'max_batsmen': 6,
    'min_bowlers': 3,
    'max_bowlers': 6,
    'wicket_keepers': 1,
    'budget_cap': 100.0
}
```

## Troubleshooting

### Common Issues

1. **Dataset Format Issues**
   ```python
   # Check column names
   print(deliveries_df.columns.tolist())
   
   # Rename columns if needed
   deliveries_df.rename(columns={'batsman': 'batter'}, inplace=True)
   ```

2. **Insufficient Training Data**
   ```
   WARNING: Insufficient training data. Using fallback statistical models.
   ```
   - Solution: Ensure dataset has sufficient historical matches (recommended: 200+ matches)

3. **Memory Issues with Large Datasets**
   ```python
   # Filter to recent seasons only
   recent_seasons = [2019, 2020, 2021, 2022, 2023]
   filtered_matches = matches_df[matches_df['season'].isin(recent_seasons)]
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/`
4. Submit pull requests with detailed descriptions

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings for all public methods
- Include unit tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IPL dataset providers
- Scikit-learn community
- Fantasy cricket community for insights
- Open source contributors

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)

## Future Enhancements

- [ ] Real-time injury/availability updates
- [ ] Weather and pitch condition integration
- [ ] Opposition-specific player performance
- [ ] Advanced captaincy analytics
- [ ] Multi-format support (T20, ODI)
- [ ] Interactive web dashboard
- [ ] API endpoints for integration
- [ ] Mobile app companion

---

**Made with love for cricket enthusiasts and data science lovers!**
