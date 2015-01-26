package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set)
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
		 * @param subset Set of nearest neighbors with their distance
		 * @return the winning class, ususally a String
		 */
		
		return "winning Class";
	}
	@Override
	protected void learnModel(List<List<Object>> traindata) {
		/*
		 * Struktur von Testdaten:
		 * Traindata: [[presbyopic, hypermetrope, yes, reduced, none], [presbyopic, hypermetrope, yes, normal, none]]
		 * Traindata: [[young, myope, no, reduced, none], [young, myope, no, normal, soft], ... ]
		 */
		//System.out.println("Traindata: " + traindata);
		this.traindata = traindata;

		
	}
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		/**
		 * Collects the votes based on an unweighted schema 
		 * @param subset Set of nearest neighbors with their distance
		 * @return Map of classes with their votes (e.g. returnValue.get("yes") are the votes for class "yes")
		 */
		
		return null;
	}
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		System.out.println("first: " + subset);
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		// TODO Auto-generated method stub
		return null;
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
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected String[] getMatrikelNumbers() {
		/* Can Diehl: 123456789
		 * Robert Pinsler: 1234567
		 * Jan Müller : 12345678
		 */
	
		String[] matrNumbers = new String[3];
		matrNumbers[0] = "123456789";
		matrNumbers[1] = "1234567";
		matrNumbers[2] = "12345678";

		
		return matrNumbers;
	}
	


}
