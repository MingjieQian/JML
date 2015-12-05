package jml;

import jml.classification.LogisticRegressionMCLBFGS;
import jml.classification.MultiClassSVM;

/***
 * JML entry for shell command lines.
 * 
 * @version 1.0 Mar. 9th, 2012
 * @version 1.1 Dec. 29th, 2012
 * @version 1.2 Jan. 1st, 2013
 * @version 1.3 Jan. 3rd, 2013
 * @version 1.4 Jan. 11th, 2013
 * 
 * @version 2.0 Feb. 4th, 2013 (feature selection package is added)
 * @version 2.1 Feb. 15th, 2013 (HMM implemented)
 * @version 2.2 Feb. 18th, 2013 (MaxEnt implemented)
 * @version 2.3 Feb. 22nd, 2013 (CRF implemented)
 * @version 2.4 Mar. 12nd, 2013 (QP problems, primal dual interior point, 
 *                               accelerated proximal gradient)
 * @version 2.5 Mar. 29th, 2013  
 * </br> &nbsp &nbsp 1. online.classification package is created. 
 * </br> &nbsp &nbsp 2. Perceptron and Winnow are implemented.
 * </br> &nbsp &nbsp 3. PCA, KernelPCA, Isomap, and LLE are implemented.
 * 
 * @version 2.6 Oct. 20th, 2013 (AdaBoost, more projection operator and 
 *                               proximal mapping classes implemented) 
 *                               
 * @version 2.7 Nov. 5th, 2013 (display with specified precision, tutorial)
 * 
 * @version 2.8 Nov. 22nd, 2013 (Robust PCA, matrix completion, svd, rank,
 *                               randperm, and linear indexing)
 * 
 * @author Mingjie Qian
 */
public class JML {

	/**
	 * Main class entry called from a command line.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {

		/*String command = "--classification -train true " +
						 "-method SVM " +
						 "-C 1 -eps 0.001 " +
						 "-model SVMModel.dat " +
						 "-trainingData LIBLINEARInputData.txt " +
						 "-testData LIBLINEARInputData.txt";
		
		args = command.split("\\s+");*/
		run(args);
		
	}
	
	/**
     * Run {@code JML} package based on arguments in a {@code String} array.
     * 
     * @param args command line arguments
     * 
     */
    private static void run(String[] args) {
    	   	
		String method = "";
		String attribute = "";
		String value = "";
		for(int i = 0; i < args.length; i++) {

			if (args[i].startsWith("--"))
				continue;
			
			if (args[i].charAt(0) != '-')
				break;

			if (++i >= args.length) {
				showUsage();
				System.exit(1);
			}				

			attribute = args[i - 1];
			value = args[i];
			
			if (attribute.equals("-method")) {
				method = value;
				break;
			}
			
		}
		
		if (method.isEmpty()) {
			showUsage();
			System.exit(1);
		}
		
		String[] subArgs = new String[args.length - 1];
		for (int i = 0; i < subArgs.length; i++) {
			subArgs[i] = args[i + 1];
		}			
		
    	if (args[0].equalsIgnoreCase("--classification")) {
    		shellClassification(subArgs, method);
    	} else if (args[0].equalsIgnoreCase("--clustering")) {
    		shellClustering(subArgs, method);
    	}

    }
    
    private static void shellClassification(String[] args, String method) {
    	
    	if (method.equalsIgnoreCase("SVM")) {
    		MultiClassSVM.run(args);
    	} else if (method.equalsIgnoreCase("LogisticRegression")) {
    		LogisticRegressionMCLBFGS.run(args);
    	}
    }
    
    private static void shellClustering(String[] args, String method) {
    	
    }
    
    /**
     * Show usage.
     */
    private static void showUsage() {
    	
    	System.out.println("Usage: java -jar JML.jar [options]\n"
				+ "options:\n"
				+ "***************************************************************\n"
				+ "--classification : Multi-class classification\n"
				+ "-train : If train a model (true) or not (false)\n"
				+ "-method : Name of a machine learning method or algorithm\n"
				+ " SVM : Multi-class SVM (linear kernel)\n"
				+ "\t-C : Parameter for the loss term\n"
				+ "\t-eps : Convergence tolerance\n"
				+ "-model : Model path\n"
				+ "-trainingData : File path for the training data with LIBSVMInput format\n"
				+ "-testData : File path for the test data with LIBSVMInput format\n"
				+ "***************************************************************\n"
				+ "--clustering : Clustering\n"
				+ "***************************************************************\n"
				+ "--regression : Regression\n"
				+ "***************************************************************\n"
				+ "--feature : Feature selection\n"
				+ "***************************************************************\n"
				+ "--subspace : Subspace learning, e.g., dimensionality reduction\n"
				+ "***************************************************************\n"
				+ "--visualization : Visualization\n"
				+ "***************************************************************\n"
				+ "--topic : Topic modeling/mining\n"
				+ "	LDA : Latent Dirichlet Analysis\n"
				+ "	L1NMF: sav norm regularized NMF\n"
				+ "-nTopic : number of topics\n"
				+ "-maxIter : maximal iteration\n"
				+ "-nTopTerm : number of top terms for each topic\n"
				+ "For LDA:\n"
				+ "-burnIn : number of burning steps for Gibbs sampling\n"
				+ "-thinInterval : number of steps between samplings\n"
				+ "-sampleLag : number of steps for sampling\n"
				);
    	
    }

}
