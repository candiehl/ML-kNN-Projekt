package tud.ke.ml.project.classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

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

	protected List<List<Object>> traindata = new ArrayList<List<Object>>();

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		/**
		 * Determines the winning class base on the subset of nearest neighbors
		 * 
		 * @param subset Set of nearest neighbors with their distance
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
		this.traindata = traindata;
		// this.traindata.addAll(traindata); // doesn't work, so probably incremental learning is not supported...
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
			Object key = instance.getA().get(this.getClassAttribute());
			if (!votesFor.containsKey(key)) {
				votesFor.put(key, 0.0); // initialize entry
			}

			votesFor.put(key, votesFor.get(key) + 1);
		}

		return votesFor;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> votesFor = new HashMap<Object, Double>();
		double weightSum = 0.0;
		double EPSILON = 0.000001; // consider as zero for numerical reasons
		double MAX_WEIGHT = 1000000; // weight for (almost) perfect match

		for (Pair<List<Object>, Double> instance : subset) {
			Object key = instance.getA().get(this.getClassAttribute());
			if (!votesFor.containsKey(key)) {
				votesFor.put(key, 0.0); // initialize entry
			}

			double distance = instance.getB();
			double weight;
			if (distance <= EPSILON) {
				weight = MAX_WEIGHT; // if distance is very small, attach some very high weight
			} else {
				weight = 1 / distance; // inverse distance weighting
			}
			votesFor.put(key, votesFor.get(key) + weight);
			weightSum += weight;
		}

		// optional normalization
		// won't change the order of the weighted instances since it's a linear transformation
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
	
	/**
	 * Randomly choose a winner after duplicating each candidate as often as the class occurs in the training set, hence guiding the selection process.
	 * 
	 * An alternative approach would be to take the frequency of the classes into account
	 * and choose the most frequent class among the possible winners or 
	 * pick the winner in a completely random fashion, i.e. each class has the same probability of being selected. 
	 * 
	 * @param winners List<Object> with all possible winners
	 * @return
	 */
	private Object breakWinnerTie(List<Object> winners) {
		// alternative approach: predict most frequent class among winners
//		if(winners.size() == 1) return winners.get(0);
//		Map<Object,Integer> classDist = this.getClassDistribution();
//		Object winningClass = null;
//		int max = 0;
//		for(Object winner : winners) {
//			if(classDist.containsKey(winner)) {
//				int count = classDist.get(winner);
//				if(count > max) {
//					max = count;
//					winningClass = winner;
//				}
//			}
//		}
		
		// alternative approach: selection is completely random, i.e. each class has the same probability of being selected
//		int index = new Random().nextInt((winners.size()));
//		Object winningClass = winners.get(index);

		
		// randomly choose a winner after duplicating each candidate as often as the class occurs in the training set
		// in case all candidates are equally frequent, it falls back to a uniformly distributed random selection
		if(winners.size() == 1) return winners.get(0);
		Map<Object,Integer> classDist = this.getClassDistribution();
		Object winningClass = null;
		for(Object winner : winners) {
			if(classDist.containsKey(winner)) {
				int count = classDist.get(winner);
				for (int i = 0; i < count-1; i++) {
					winners.add(winner);
				}
			}
		}
		
		int index = new Random().nextInt((winners.size()));
		winningClass = winners.get(index);
		
		return winningClass;
	}
	
	private Map<Object,Integer> getClassDistribution() {
		Map<Object,Integer> classDist = new HashMap<Object,Integer>();
		for (List<Object> trainInstance : this.traindata) {
			Object classAtt = trainInstance.get(this.getClassAttribute());
			if(!classDist.containsKey(classAtt)) {
				classDist.put(classAtt, 0);
			}
			classDist.put(classAtt, classDist.get(classAtt)+1);
		}
		return classDist;
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		List<Pair<List<Object>, Double>> nearest = new ArrayList<Pair<List<Object>, Double>>();

		if(!checkInput(testdata)) throw new IllegalArgumentException("Malformed input");
		
		// assumption: input data should be immutable, i.e. original training and test data are retained
		List<List<Object>> trainInstances = new ArrayList<List<Object>>();
		trainInstances.addAll(traindata);
		List<Object> testInstance = new ArrayList<Object>();
		testInstance.addAll(testdata);
		
		if (isNormalizing()) {
			boolean scaled = false;
			if (scaling == null || translation == null) {
				double[][] factors = normalizationScaling();
				scaling = factors[0];
				translation = factors[1];
			} else scaled = true;			
			
			for(int i=0; i < scaling.length ; i++){
				if(scaling[i] != 0.0d) { // excludes constant numeric attributes (MAX==MIN) as well as nominal attributes (initialized with translation=0.0) 
					testInstance.set(i, ((double)testdata.get(i) - translation[i]) / scaling[i]);
					// in case the class attribute is numeric (which it shouldn't), do not normalize it
					if(i == this.getClassAttribute()) continue;
					if(!scaled) {
						for (List<Object> trainInstance : trainInstances) {
							trainInstance.set(i, ((double)trainInstance.get(i) - translation[i]) / scaling[i]);
						}
					}
				}
			}
		}
		
		// usually, we'd expect that the training set has one more attribute than the test instance
		// if not, the test instance probably comes with its own class label, which we should not take into account
		if(traindata.get(0).size() == testdata.size()) {
			testInstance.remove(this.getClassAttribute());
		} 
		
		switch (this.getMetric()) {
		case 0: { // Manhattan distance
			for (List<Object> trainInstance : trainInstances) {
				List<Object> trainInst = new ArrayList<Object>();
				trainInst.addAll(trainInstance);
				trainInst.remove(this.getClassAttribute());
				// store complete instance, but exclude class attribute from distance calculation
				nearest.add(new Pair<List<Object>, Double>(trainInstance, determineManhattanDistance(trainInst, testInstance)));
			}
			break;
		}
		case 1: { // Euclidean distance
			for (List<Object> trainInstance : trainInstances) {
				List<Object> trainInst = new ArrayList<Object>();
				trainInst.addAll(trainInstance);
				trainInst.remove(this.getClassAttribute());
				// store complete instance, but exclude class attribute from distance calculation
				nearest.add(new Pair<List<Object>, Double>(trainInstance, determineEuclideanDistance(trainInst, testInstance)));
			}
			break;
		}
		default: throw new IllegalArgumentException("Distance metric not supported");
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
	
	/**
	 * Perform some sanity checks on the input data.
	 * @param testdata
	 * @return
	 */
	private boolean checkInput(List<Object> testdata) {
		if(this.traindata == null || testdata == null || this.traindata.size() == 0) return false;
		if(this.traindata.get(0).size() < testdata.size() || this.traindata.get(0).size() == 0 || testdata.size() == 0) return false;
		// add more checks here...
		return true;		
	}


	/**
	 * Choose the nearest neighbors.
	 * 
	 * For tie breaks increase k until tie is resolved or k==n (subsequent ties are broken after the voting process)
	 * 
	 * @param nearest all neighbors sorted by distance
	 * @param k
	 * @return
	 */
	private List<Pair<List<Object>, Double>> breakNearestTie(
			List<Pair<List<Object>, Double>> nearest, int k) {

		while((k+1) < nearest.size() && nearest.get(k).getB() == nearest.get(k+1).getB()) {
			k += 1;
		}
		
		return nearest.subList(0, Math.min(nearest.size(), k)); // if k is larger than the number of instances, return all instances
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		
		// assumption: use 0/1 distance for nominal attributes		
		double distance = 0.0;
		for (int i = 0; i < instance1.size(); i++) {
			Object attribute1 = instance1.get(i);
			Object attribute2 = instance2.get(i);
			if(attribute1 instanceof String){ // attribute is nominal				
				if(!attribute1.equals(attribute2)) {
					distance += 1.0;
				}
				
			} else if ((attribute1 instanceof Double) || (attribute1 instanceof Integer)) { // attribute is continuous				
				distance += Math.abs((Double)attribute1 - (Double)attribute2);
			}	
		}
		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {

		// assumption: use 0/1 distance for nominal attributes
		double distance = 0.0;
		for (int i = 0; i < instance1.size(); i++) {
			Object attribute1 = instance1.get(i);
			Object attribute2 = instance2.get(i);
			if(attribute1 instanceof String){ // attribute is nominal
				if(!attribute1.equals(attribute2)){
					distance += 1.0;
				}
						
			} else if ((attribute1 instanceof Double) || (attribute1 instanceof Integer)) { // attribute is continuous
				distance += (Math.pow(Math.abs((Double)attribute1 - (Double)attribute2),2));
			}
		}
		
		return Math.sqrt(distance);
	}
	
	@Override
	protected double[][] normalizationScaling() {
		// use range normalization, i.e. (x_i - min(x)) / (max(x)-min(x))
		// alternatively, z-normalization could be used
		int size = this.traindata.get(0).size();
		double[][] factors = new double[2][size];
		double[] maxFactors = new double[size];
		double[] minFactors = new double[size];
		for (List<Object> trainInstance : traindata) {
			for (int i = 0; i < trainInstance.size(); i++) { 
				Object attribute = trainInstance.get(i);
				if((attribute instanceof Double) || (attribute instanceof Integer)) {
					double numAttribute = (double)attribute;
					if (minFactors[i] > numAttribute || minFactors[i] == 0.0d){
						minFactors[i] = numAttribute;
					}
					if (maxFactors[i] < numAttribute || minFactors[i] == 0.0d){
						maxFactors[i] = numAttribute;
					}
				}
			}
		}
		
		for (int i = 0; i < size; i++) {
			factors[0][i] = maxFactors[i] - minFactors[i]; // scaling
			factors[1][i] = minFactors[i]; // translation
		}
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