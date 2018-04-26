# pgm-oil

We propose using Probabilistic Graphical Models such as Belief Networks and Hidden Markov Models to construct a global macro trading strategy of the Crude Oil Markets.

## Data Sources

Our Proof-of-Concept Crude Oil Trading quantitative model retrieves time-series data from a number of different government open-data facilities such as the [Energy Infromation Adminstration](https://www.eia.gov/) and the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/).

We use the EIA for understanding the physical market factors that affect the price of crude oil. Such factors include, but are not limited to, the Consumption (Demand) from OECD and non-OECD countries, the Supply (Production) from OPEC and non-OPEC countries, the U.S. Import and Export of Crude Oil, and the Strategic Petroleum Reserve (SPR) withdrawals.

We use the FRED for understanding how the macroeconomic factors affect the price of Crude Oil. Such factors include, but are not limited to, the Industrial Production Index, Interest Rates, Consumer Price Index (CPI), Producer Price Index (PPI).

## Graphical Models

### Hidden Markov Models (HMMs)

We use Hidden Markov Models in order to detect underlying regimes of the time-series data so that we can discretise the continuous time-series data. We use the Baum-Welch algorithm for learning the HMMs, and we use the Viterbi Algorithm to find the sequence of hidden states (i.e. the regimes) given the observed states (i.e. monthly differences) of the time-series.

We have used [hmms](https://github.com/lopatovsky/HMMs) by Lukas Lopatovsky for implementing HMMs.

### Belief Networks

We use Belief Networks so that we can analyse the probability of a regime in the Crude Oil given a set of different regimes in the macroeconomic factors as evidence. We chose a *Greedy Hill Climbing algorithm* to learn the Belief Network, and then learned the Parameters using *Bayesian Estimation* using a K2 prior. We then performed inferences on the Belief Networks to obtain a forecast of the crude oil markets, and then tested the forecast on real data.

We have used [pgmpy](https://github.com/pgmpy) by Ankur Ankan and Abinash Panda for implementing Belief Networks.

The implementation is given in the [the Jupyter notebook](https://github.com/DanishAmjadAlvi/pgm-oil/blob/master/notebook/Implementation.ipynb), covering most of the development stages.

## Acknowlegements

This project was the final year project submitted by Danish A. Alvi, an undergraduate at University College London, Department of Computer Science as part of his completion of Bachelors in Computer Science.

We are grateful to Ankur Ankan, Abinash Panda, and Khali Bartan for the support given in this project with pgmpy.

## Disclaimer

This project is only an experiment and a Proof-of-Concept of using Bayesian Networks to forecast the price of oil using graphical models. We do not accept any responsibility for any losses incurred with the use of the model.
