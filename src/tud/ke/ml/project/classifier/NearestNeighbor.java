package tud.ke.ml.project.classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but
 * probably not set)
 * 
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {

	protected double[] scaling;
	protected double[] translation;

	protected List<List<Object>> traindata;

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		/**
		 * Determines the winning class base on the subset of nearest neighbors
		 * 
		 * @param subset
		 *            Set of nearest neighbors with their distance
		 * @return the winning class, ususally a String
		 */

		Map<Object, Double> votesFor;
		if (isInverseWeighting()) {
			votesFor = getWeightedVotes(subset);
		} else {
			votesFor = getUnweightedVotes(subset);
		}

		return (getWinner(votesFor));
	}

	@Override
	protected void learnModel(List<List<Object>> traindata) {
		System.out.println("Traindata: " + traindata);
		this.traindata = traindata;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		/**
		 * Collects the votes based on an unweighted schema
		 * 
		 * @param subset
		 *            Set of nearest neighbors with their distance
		 * @return Map of classes with their votes (e.g. returnValue.get("yes")
		 *         are the votes for class "yes")
		 */

		Map<Object, Double> votesFor = new HashMap<Object, Double>();
		for (Pair<List<Object>, Double> instance : subset) {
			Object key = instance.getA();
			if (!votesFor.containsKey(key)) {
				votesFor.put(key, 0.0);
			}

			votesFor.put(key, votesFor.get(key) + 1);
		}

		return votesFor;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		System.out.println("first: " + subset);
		Map<Object, Double> votesFor = new HashMap<Object, Double>();
		double weightSum = 0.0;
		double EPSILON = 0.000001; // consider as zero for numerical reasons
		double MAX_WEIGHT = 1000000; // weight for (almost) perfect match

		for (Pair<List<Object>, Double> instance : subset) {
			Object key = instance.getA();
			if (!votesFor.containsKey(key)) {
				votesFor.put(key, 0.0);
			}

			double distance = instance.getB();
			double weight;
			if (distance <= EPSILON) {
				weight = MAX_WEIGHT;
			} else {
				weight = 1 / distance;
			}
			votesFor.put(key, votesFor.get(key) + weight);
			weightSum += weight;
		}

		// optional normalization
		if (weightSum >= EPSILON) {
			for (Entry<Object, Double> entry : votesFor.entrySet()) {
				votesFor.put(entry.getKey(), entry.getValue() / weightSum);
			}
		}

		return votesFor;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		List<Object> winners = new ArrayList<Object>();
		double max = Collections.max(votesFor.values());

		for (Entry<Object, Double> entry : votesFor.entrySet()) {
			if (max == entry.getValue()) {
				winners.add(entry.getKey());
			}
		}
		return breakWinnerTie(winners);
	}

	private Object breakWinnerTie(List<Object> winners) {
		return winners.get(0); // TODO: implement proper selection strategy
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		List<Pair<List<Object>, Double>> nearest = new ArrayList<Pair<List<Object>, Double>>();

		if (isNormalizing()) {
			double[][] factors = normalizationScaling();
			scaling = factors[0];
			translation = factors[1];
			// TODO: implement
		}

		switch (this.getMetric()) {
		case 0: { // Manhattan distance
			for (List<Object> trainInstance : traindata) {
				nearest.add(new Pair<List<Object>, Double>(trainInstance,
						determineManhattanDistance(trainInstance, testdata)));
			}
			break;
		}
		case 1: { // Euclidean distance
			for (List<Object> trainInstance : traindata) {
				nearest.add(new Pair<List<Object>, Double>(trainInstance,
						determineEuclideanDistance(trainInstance, testdata)));
			}
			break;
		}
		// default: do nothing
		}

		Collections.sort(nearest, new Comparator<Pair<List<Object>, Double>>() {
			@Override
			public int compare(Pair<List<Object>, Double> element1,
					Pair<List<Object>, Double> element2) {
				return element1.getB().compareTo(element2.getB());
			}
		});

		return breakNearestTie(nearest, this.getkNearest());
	}

	private List<Pair<List<Object>, Double>> breakNearestTie(
			List<Pair<List<Object>, Double>> nearest, int k) {
		
		return nearest.subList(0, Math.min(nearest.size(), k)); // TODO: implement proper
														// selection strategy
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	protected double[][] normalizationScaling() {
		double[][] factors = new double[2][this.traindata.get(0).size()];
		// TODO: implement
		return factors;
	}

	@Override
	protected String[] getMatrikelNumbers() {
		String[] matrNumbers = new String[3];
		matrNumbers[0] = "1464470";
		matrNumbers[1] = "2308991";
		matrNumbers[2] = "2840608";

		return matrNumbers;
	}
}
