package jml.topics;

import org.apache.commons.math.linear.RealMatrix;

/**
 * Abstract super class for all topic models.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan 3rd, 2013
 */
public abstract class TopicModel {
	
	/**
	 * A V-by-N matrix with each column being a term-count array.
	 */
	protected RealMatrix dataMatrix;
	
	/**
	 * A V-by-K matrix, where each column is a topic vector
	 * represented by a vector of weights.
	 */
	protected RealMatrix topicMatrix;
	
	/**
	 * A N-by-K matrix holding document--topic associations.
	 */
	protected RealMatrix indicatorMatrix;
	
	/**
	 * Number of topics.
	 */
	public int nTopic;
	
	/**
	 * Default constructor for this topic model.
	 */
	public TopicModel() {
		System.err.println("Number of topics undefined!");
		System.exit(1);
	}
	
	/**
	 * Constructor for this topic model given the number of topics
	 * for a corpus.
	 * 
	 * @param nTopic number of topics to fit a corpus
	 * 
	 */
	public TopicModel(int nTopic) {
		if (nTopic < 1) {
			System.err.println("Number of topics less than one!");
			System.exit(1);
		}
		this.nTopic = nTopic;
	}
	
	/**
	 * Read corpus from a V-by-N document-term-count matrix
	 * for this topic model.
	 * 
	 * @param dataMatrix a V-by-N document-term-count matrix
	 * 
	 */
	public void readCorpus(RealMatrix dataMatrix) {
		this.dataMatrix = dataMatrix;
	}
	
	/**
	 * Get the V-by-K topic matrix for this topic model.
	 * 
	 * @return a V-by-K topic matrix
	 * 
	 */
	public RealMatrix getTopicMatrix() {
		return topicMatrix;
	}
	
	/**
	 * Get the N-by-K topic assignment matrix for this topic model.
	 * 
	 * @return an N-by-K topic indicator matrix
	 * 
	 */
	public RealMatrix getIndicatorMatrix() {
		return indicatorMatrix;
	}

	/**
	 * Train this topic model to fit the given corpus.
	 */
	public abstract void train();
	
}
