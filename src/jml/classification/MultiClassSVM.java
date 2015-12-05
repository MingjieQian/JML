package jml.classification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

import jml.data.Data;
import jml.matlab.Matlab;
import static jml.matlab.Matlab.display;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;


import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.InvalidInputDataException;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

/***
 * Multi-class SVM using LIBLINEAR library.
 * 
 * @author Mingjie Qian
 * @version 1.0, Jan. 3rd, 2013
 */
public class MultiClassSVM extends Classifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 6720660840680032348L;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
			
		double[][] data = { {3.5, 4.4, 1.3},
							{5.3, 2.2, 0.5},
							{0.2, 0.3, 4.1},
							{-1.2, 0.4, 3.2} };
		
		double[][] labels = { {1, 0, 0},
				  			  {0, 1, 0},
				  			  {0, 0, 1} };
		
		double C = 1;
		double eps = 0.01;
		Classifier multiClassSVM = new MultiClassSVM(C, eps);
		multiClassSVM.feedData(data);
		multiClassSVM.feedLabels(labels);
		multiClassSVM.train();
		
		RealMatrix Y_pred = multiClassSVM.predictLabelMatrix(data);
		display(Y_pred);
		
		System.out.println("Projection matrix:");
		Matlab.printMatrix(multiClassSVM.getProjectionMatrix());
		
		System.out.println("Predicted label score matrix:");
		display(multiClassSVM.predictLabelScoreMatrix(data));
		
		// Get elapsed time in seconds
		long start = System.currentTimeMillis();

		String trainDataFilePath = "heart_scale";
		C = 1;
		eps = 0.01;
		MultiClassSVM MultiClassSVM = new MultiClassSVM(C, eps);
		MultiClassSVM.feedProblem(trainDataFilePath);
		MultiClassSVM.train();
		System.out.println("Using Linear.predict method:");
		int[] pre_label = MultiClassSVM.predict2(MultiClassSVM.features, MultiClassSVM.labels);
		getAccuracy(pre_label, MultiClassSVM.labels);
		System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);
		
		System.out.println("Using our predict method:");
		RealMatrix XTrain = features2Matrix(MultiClassSVM.features);
		int[] pre_label1 = MultiClassSVM.predict(XTrain);
		getAccuracy(pre_label1, MultiClassSVM.labels);
		System.out.println("There are " + pre_label.length + " samples predicted.\n");
		
		System.out.println("Using predict(Feature[][]) method:");
		pre_label1 = MultiClassSVM.predict(matrix2Features(XTrain));
		getAccuracy(pre_label1, MultiClassSVM.labels);
		
		System.out.format("Elapsed time: %.2f seconds.%n", (System.currentTimeMillis() - start) / 1000F);

		/********************************************/
		Classifier MCSVM = new MultiClassSVM(C, eps);
		String dataMatrixFilePath = "CNN - DocTermCount.txt";
		RealMatrix X = Data.loadMatrixFromDocTermCountFile(dataMatrixFilePath);
		MCSVM.feedData(X);
		String labelMatrixFilePath = "GroundTruth.txt";
		RealMatrix Y = Data.loadMatrix(labelMatrixFilePath);
		String sparseLabelMatrixFilePath = "GroundTruthSparse.txt";
		Data.saveSparseMatrix(sparseLabelMatrixFilePath, Y);
		Y = Data.loadMatrix(sparseLabelMatrixFilePath);
		MCSVM.feedLabels(Y);
		MCSVM.train();
		
		pre_label = MCSVM.predict(X);
		getAccuracy(pre_label, MCSVM.labels);
		
		String modelFilePath = "MCSVMModel";
		MCSVM.saveModel(modelFilePath);
		
		/*RealMatrix Yt = MCSVM.predictLabelMatrix(X);
		Matlab.printMatrix(Yt);*/
		
		/********************************************/
		Classifier MCSVM2 = new MultiClassSVM();
		MCSVM2.loadModel(modelFilePath);
		int[] pre_label2 = MCSVM2.predict(X);
		int[] labelIndices = labelScoreMatrix2LabelIndexArray(Y);
		getAccuracy(pre_label2, labelIndices);
		System.out.println("There are " + pre_label2.length + " samples predicted.\n");

		System.out.println("Using predict(Feature[][]) method:");
		int[] pre_label3 = ((MultiClassSVM)MCSVM2).predict(matrix2Features(X));
		getAccuracy(pre_label3, labelIndices);
		
		System.out.println("Mission complete!");
		
	}
	
	/**
     * Run this module with arguments in a {@code String} array.
     * 
     * @param args command line arguments
     * 
     */
	public static void run(String[] args) {
		
		double C = 1;
		double eps = 0.001;
		String modelFilePath = "";
		String trainingDataFilePath = "";
		String testDataFilePath = "";
		boolean doesTrain = false;
		
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
			
			if (attribute.equals("-C")) {
				C = Double.parseDouble(value);
			} else if (attribute.equals("-eps")) {
				eps = Double.parseDouble(value);
			} else if (attribute.equals("-model")) {
				modelFilePath = value;
			} else if (attribute.equals("-trainingData")) {
				trainingDataFilePath = value;
			} else if (attribute.equals("-testData")) {
				testDataFilePath = value;
			} else if (attribute.equals("-train")) {
				doesTrain = Boolean.parseBoolean(value);
			}
			
		}
		
		if (trainingDataFilePath.isEmpty() && testDataFilePath.isEmpty()) {
			showUsage();
			System.exit(1);
		}
		
		if (doesTrain) {
			if (modelFilePath.isEmpty()) {
				System.err.println("Model file path is empty.");
				showUsage();
				System.exit(1);
			}
			Problem problem = MultiClassSVM.readProblem(trainingDataFilePath);
			RealMatrix X = MultiClassSVM.features2Matrix(problem.x);
			
			Classifier multiClassSVM = new MultiClassSVM(C, eps);
			multiClassSVM.feedData(X);
			multiClassSVM.feedLabels(problem.y);
			multiClassSVM.train();
			multiClassSVM.saveModel(modelFilePath);
			
			if (!testDataFilePath.isEmpty()) {
				problem = MultiClassSVM.readProblem(trainingDataFilePath);
				RealMatrix Xt = MultiClassSVM.features2Matrix(problem.x);
				int[] pred_labels = multiClassSVM.predict(Xt);
				for (int i = 0; i < pred_labels.length; i++) {
					System.out.format("Doc %d: y_pred: %s\n",
							i, pred_labels[i]);
				}
				System.out.println();
			}
		} else {
			Problem problem = MultiClassSVM.readProblem(trainingDataFilePath);
			RealMatrix Xt = MultiClassSVM.features2Matrix(problem.x);
			Classifier multiClassSVM = new MultiClassSVM();
			multiClassSVM.loadModel(modelFilePath);
			int[] pred_labels = multiClassSVM.predict(Xt);
			int[] IDlabelMap = ((MultiClassSVM)multiClassSVM).model.getLabels();
			for (int i = 0; i < pred_labels.length; i++) {
				System.out.format("Doc %d: y_pred: %s\n", i, IDlabelMap[pred_labels[i]]);
			}
			System.out.println();
		}
		
	}

	/**
     * Show usage.
     */
	private static void showUsage() {
		
	}

	/**
	 * A {@code Problem} object for this MCSVM classifier.
	 */
	Problem problem;
	
	/**
	 * Feature 2D array, indices start from 1.
	 */
	Feature[][] features;
	
	/**
	 * Parameter for loss term of linear multi-class SVM.
	 */
	double C;
	
	/**
	 * Convergence tolerance.
	 */
	double eps;
	
	/**
	 * A {@code Parameter} instance for linear multi-class SVM.
	 */
	Parameter parameter;

	/**
	 * Dummy feature, aiming to remove the equality constraint for
	 * the dual problem of SVM.
	 */
	private double bias;
	
	/**
	 * SVM model.
	 */
	Model model;
	
	/**
	 * Maximal feature index, not including the bias feature.
	 */
	// int max_index;

	public MultiClassSVM(double C, double eps) {
		super();
		this.bias = 1;
		this.C = C;
		this.eps = eps;
		features = null;
		parameter = new Parameter(SolverType.MCSVM_CS, C, eps);
		model = null;
	}

	public MultiClassSVM() {
	}

	/**
	 * Feed a problem from a string array.
	 * @param feaArray a string array of which each element is a
	 *                 index value feature representation for an 
	 *                 example with the LIBSVM input data format
	 */
	public void feedProblem(ArrayList<String> feaArray) {
		feedProblem(readProblemFromStringArray(feaArray));
	}
	
	/**
	 * Feed a problem from a file with the LIBLINEAR input data format.
	 * 
	 * @param filePath file path for a file with the LIBLINEAR input
	 *                 data format
	 * 
	 */
	public void feedProblem(String filePath) {
		Problem problem = readProblemFromFile(new File(filePath), bias);
		feedProblem(problem);
		System.out.println("Problem loaded from file " + filePath);
	}

	/**
	 * Feed a problem for this SVM classifier.
	 * 
	 * @param problem a {@code Problem} object
	 * 
	 */
	public void feedProblem(Problem problem) {
		this.problem = problem;
		features = problem.x;
		super.feedData(features2Matrix(features));
		labels = problem.y;
		feedLabels(labels);
	}

	/**
	 * {@inheritDoc}
	 * For MultiClassSVM, the original data matrix should not contain
	 * the bias dummy feature.
	 * 
	 * @param X original data matrix without bias dummy features
	 * 
	 */
	@Override
	public void feedData(RealMatrix X) {
		features = matrix2Features(X, bias);
		super.feedData(X);
	}

	/**
	 * Get the maximal feature index not including the bias feature.
	 * 
	 * @param features a 2D feature array
	 * 
	 * @param bias dummy bias feature value
	 * 
	 * @return maximal raw feature index not including
	 *         the bias feature
	 * 
	 */
	public static int getMaxRawFeatureIndex(Feature[][] features, double bias) {
		int maxIndex = 0;
		for (int i = 0; i < features.length; i++) {
			int j = 0;
			j = bias >= 0 ? features[i].length - 1: features[i].length;
			maxIndex = Math.max(maxIndex, features[i][j - 1].getIndex());
		}
		return maxIndex;	
	}
	
	@Override
	public void train() {
		
		if (problem == null) {
			problem = new Problem();
			problem.bias = bias;
			problem.l = features.length;
			problem.x = features;
			problem.y = labels;
			problem.n = getMaxRawFeatureIndex(features, bias);
			nFeature = problem.n;
			if (bias >= 0) {
				problem.n++;
			}
		}
		
		model = Linear.train(problem, parameter);
		reconstructProjectionMatrix();
	
	}

	@Deprecated
	public static int[] predict2(Model model, Feature[][] data, int[] labels) {

		int N = data.length;
		int[] pre_label = new int[N];

		for ( int i = 0; i < N; i ++ ) {
			pre_label[i] = Linear.predict(model, data[i]);
		}

		if (labels != null) {
			int cnt_correct = 0;
			for ( int i = 0; i < N; i ++ ) {
				if ( pre_label[i] == labels[i] )
					cnt_correct ++;
			}
			double accuracy = (double)cnt_correct / (double)N;
			System.out.println(String.format("Accuracy: %.2f%%\n", accuracy * 100));
		}

		return pre_label;

	}
	
	@Deprecated
	public int[] predict2(Feature[][] data, int[] labels) {

		int N = data.length;
		int[] pre_label = new int[N];
		double[] values = new double[nClass];
		for ( int i = 0; i < N; i ++ ) {
			pre_label[i] = Linear.predictValues(model, data[i], values);
		}

		if (labels != null) {
			int cnt_correct = 0;
			for ( int i = 0; i < N; i ++ ) {
				if ( pre_label[i] == labels[i] )
					cnt_correct ++;
			}
			double accuracy = (double)cnt_correct / (double)N;
			System.out.println(String.format("Accuracy: %.2f%%\n", accuracy * 100));
		}

		return pre_label;

	}

	@Deprecated
	public int[] predict2(RealMatrix Xt) {
		return this.predict2(Xt, null);
	}
	
	@Deprecated
	public int[] predict2(RealMatrix Xt, int[] labels) {
		Feature[][] x = matrix2Features(Xt, bias);
		return this.predict2(x, labels);
	}
	
	/*@Deprecated
	public RealMatrix predictLabelMatrix2(RealMatrix Xt) {
		return predictLabelMatrix2(Xt, null);
	}*/
	
	/*@Deprecated
	public RealMatrix predictLabelMatrix2(RealMatrix Xt, int[] labels) {
		Feature[][] x = matrix2Features(Xt, bias);
		int[] pred_labels = this.predict2(x, labels);
		return labelArray2LabelMatrix(pred_labels, nClass);
	}*/

	/**
	 * Data format (index starts from 1):<p/>
	 * <pre>
	 * +1 1:0.708333 2:1 3:1 4:-0.320755 5:-0.105023 6:-1 7:1 8:-0.419847 9:-1 10:-0.225806 12:1 13:-1 
	 * -1 1:0.583333 2:-1 3:0.333333 4:-0.603774 5:1 6:-1 7:1 8:0.358779 9:-1 10:-0.483871 12:-1 13:1 
	 * </pre>
	 * @param file
	 * 
	 * @param bias
	 * 
	 * @return a {@code Problem} instance holding features and labels
	 * 
	 */
	public static Problem readProblemFromFile(File file, double bias) {
		try {
			return Problem.readFromFile(file, bias);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	/**
	 * Read a problem from a file with the LIBSVM data format.
	 * 
	 * @param filePath file path
	 * 
	 * @return a {@code Problem} instance holding features and labels
	 * 
	 */
	public static Problem readProblem(String filePath) {
		return readProblemFromFile(new File(filePath), 1.0);
	}
	
	/**
	 * Convert original data matrix into Feature[][] including bias features.
	 * 
	 * Bias is always set to 1 automatically.
	 * 
	 * @param A original data matrix with each column being a sample
	 * 
	 * @return a 2D feature array
	 * 
	 */
	public static Feature[][] matrix2Features(RealMatrix A) {
		return matrix2Features(A, 1);
	}

	/**
	 * Convert original data matrix into Feature[][] including bias features
	 * if bias is nonnegative.
	 * 
	 * @param A original data matrix with each column being a sample
	 * 
	 * @param bias dummy bias feature, i.e., 1
	 * 
	 * @return a 2D feature array
	 * 
	 */
	public static Feature[][] matrix2Features(RealMatrix A, double bias) {

		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		ArrayList<Feature> featureArr = null;
		List<Feature[]> vx = new ArrayList<Feature[]>();

		// int sampleID = 0;

		int max_index = 0;
		int m = 0;
		int index = -1;
		double value = 0;
		for (int j = 0; j < nCol; j++) {
			// sampleID = j + 1;
			featureArr = new ArrayList<Feature>();
			for (int i = 0; i < nRow; i++) {
				index = i + 1;
				value = A.getEntry(i, j);
				if (value != 0) {
					featureArr.add(new FeatureNode(index, value));
				}
			}
			m = featureArr.size();
			Feature[] x;
			if (bias >= 0) {
				x = new Feature[m + 1];
			} else {
				x = new Feature[m];
			}
			Iterator<Feature> iter = featureArr.iterator();
			int k = 0;
			while(iter.hasNext()) {
				x[k] = iter.next();
				k++;
			}
			vx.add(x);
			if (m > 0) {
				max_index = Math.max(max_index, x[m - 1].getIndex());
			}
		}

		Problem prob = new Problem();
		prob.bias = bias;
		prob.l = vx.size();
		prob.n = max_index;
		if (bias >= 0) {
			prob.n++;
		}
		prob.x = new Feature[prob.l][];
		for (int i = 0; i < prob.l; i++) {
			prob.x[i] = vx.get(i);
			if (bias >= 0) {
				assert prob.x[i][prob.x[i].length - 1] == null;
				prob.x[i][prob.x[i].length - 1] = new FeatureNode(max_index + 1, bias);
			}
		}

		/*prob.y = new int[prob.l];
		for (int i = 0; i < prob.l; i++)
			prob.y[i] = vy.get(i);*/

		return prob.x;

	}
	
	/**
	 * Read a problem from a string array with default bias feature (1.0).
	 *  
	 * @param feaArray a {@code ArrayList<String>}, each element
	 *                 is a string with LIBSVM data format
	 *                 
	 * @return a {@code Problem} instance holding features and labels
	 * 
	 */
	public static Problem readProblemFromStringArray(ArrayList<String> feaArray) {
		return readProblemFromStringArray(feaArray, 1);
	}

	/**
	 * Read a problem from a string array.
	 * 
	 * @param feaArray a {@code ArrayList<String>}, each element
	 *                 is a string with LIBSVM data format
	 * 
	 * @param bias a real number to append the feature vector e.g., 1
	 * 
	 * @return a {@code Problem} instance holding features and labels
	 * 
	 */
	public static Problem readProblemFromStringArray(ArrayList<String> feaArray, double bias) {

		List<Integer> vy = new ArrayList<Integer>();
		List<Feature[]> vx = new ArrayList<Feature[]>();
		int max_index = 0;

		int lineNr = 0;

		Iterator<String> iter = feaArray.iterator();
		while (iter.hasNext()) {
			String line = iter.next();
			if (line == null) break;
			lineNr++;

			StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
			String token;
			token = st.nextToken();

			try {
				vy.add(atoi(token));
			} catch (NumberFormatException e) {
				System.err.println(String.format("invalid label: %s at line %d", token, lineNr));
			}

			int m = st.countTokens() / 2;
			Feature[] x;
			if (bias >= 0) {
				x = new Feature[m + 1];
			} else {
				x = new Feature[m];
			}
			int indexBefore = 0;
			for (int j = 0; j < m; j++) {

				token = st.nextToken();
				int index = 0;
				try {
					index = atoi(token);
				} catch (NumberFormatException e) {
					System.err.println(String.format("invalid index: %d at line %d", index, lineNr));
					System.exit(1);
				}

				// assert that indices are valid and sorted
				if (index < 0)  {
					System.err.println(String.format("invalid index: %d at line %d", index, lineNr));
					System.exit(1);
				}
				if (index <= indexBefore) {
					System.err.println(String.format("indices must be sorted in ascending order at line %d", lineNr));
					System.exit(1);
				}

				indexBefore = index;

				token = st.nextToken();
				try {
					double value = atof(token);
					x[j] = new FeatureNode(index, value);
				} catch (NumberFormatException e) {
					System.err.println(String.format("invalid value: %f at line ", token, lineNr));
					System.exit(1);
				}
			}
			if (m > 0) {
				max_index = Math.max(max_index, x[m - 1].getIndex());
			}

			vx.add(x);
		}

		return constructProblem(vy, vx, max_index, bias);
	}

	private static Problem constructProblem(List<Integer> vy, List<Feature[]> vx, int max_index, double bias) {
		Problem prob = new Problem();
		prob.bias = bias;
		prob.l = vy.size();
		prob.n = max_index;
		if (bias >= 0) {
			prob.n++;
		}
		prob.x = new Feature[prob.l][];
		for (int i = 0; i < prob.l; i++) {
			prob.x[i] = vx.get(i);

			if (bias >= 0) {
				assert prob.x[i][prob.x[i].length - 1] == null;
				prob.x[i][prob.x[i].length - 1] = new FeatureNode(max_index + 1, bias);
			}
		}

		prob.y = new int[prob.l];
		for (int i = 0; i < prob.l; i++)
			prob.y[i] = vy.get(i);

		return prob;
	}

	/**
	 * @param s the string to parse for the double value
	 * @throws IllegalArgumentException if s is empty or represents NaN or Infinity
	 * @throws NumberFormatException see {@link Double#parseDouble(String)}
	 */
	static double atof(String s) {
		if (s == null || s.length() < 1) throw new IllegalArgumentException("Can't convert empty string to integer");
		double d = Double.parseDouble(s);
		if (Double.isNaN(d) || Double.isInfinite(d)) {
			throw new IllegalArgumentException("NaN or Infinity in input: " + s);
		}
		return (d);
	}

	/**
	 * @param s the string to parse for the integer value
	 * @throws IllegalArgumentException if s is empty
	 * @throws NumberFormatException see {@link Integer#parseInt(String)}
	 */
	static int atoi(String s) throws NumberFormatException {
		if (s == null || s.length() < 1) throw new IllegalArgumentException("Can't convert empty string to integer");
		// Integer.parseInt doesn't accept '+' prefixed strings
		if (s.charAt(0) == '+') s = s.substring(1);
		return Integer.parseInt(s);
	}
	
	/**
	 * Convert features to original data matrix.
	 * Features must always have bias of 1.
	 * 
	 * @param features a 2D feature array
	 * 
	 * @return a real matrix
	 * 
	 */
	public static RealMatrix features2Matrix(Feature[][] features) {
		return features2MatrixWithoutBias(features);
	}
	
	/**
	 * Convert features to matrix including bias features if bias is nonnegative.
	 * Feature indices start from 1.
	 * 
	 * @param features a 2D feature array
	 * 
	 * @param bias dummy bias feature
	 * 
	 * @return a real matrix
	 * 
	 */
	public static RealMatrix features2Matrix(Feature[][] features, double bias) {
		
		int maxIndex = 0;
		for (int i = 0; i < features.length; i++) {
			int j = 0;
			j = bias >= 0 ? features[i].length - 1: features[i].length;
			maxIndex = Math.max(maxIndex, features[i][j - 1].getIndex());
		}
		if (bias >= 0) {
			maxIndex++;
		}
		
		RealMatrix res = new OpenMapRealMatrix(maxIndex, features.length);
		int index = 0;
		double value = 0;
		for (int j = 0; j < features.length; j++) {
			for (int i = 0; i < features[j].length; i++) {
				index = features[j][i].getIndex() - 1;
				value = features[j][i].getValue();
				res.setEntry(index, j, value);
			}
		}
		return res;
		
	}
	
	/**
	 * Convert features to original data matrix.
	 * 
	 * Features must always have bias of 1.
	 * 
	 * @param features a 2D feature array
	 * 
	 * @return a real matrix
	 * 
	 */
	public static RealMatrix features2MatrixWithoutBias(Feature[][] features) {
		return features2MatrixWithoutBias(features, 1);
	}
	
	/**
	 * Convert features to matrix excluding bias features if bias is nonnegative.
	 * 
	 * @param features a 2D feature array
	 * 
	 * @param bias dummy bias feature
	 * 
	 * @return a real matrix
	 * 
	 */
	public static RealMatrix features2MatrixWithoutBias(Feature[][] features, double bias) {
		
		RealMatrix res = features2Matrix(features, bias);
		if (bias >= 0) {
			int nRow = res.getRowDimension();
			int nCol = res.getColumnDimension();
			res = res.getSubMatrix(0, nRow - 2, 0, nCol - 1);
		}
		return res;
		
	}
	
	/**
	 * Predict label matrix for column vectors of an original data matrix.
	 * The input matrix must not include bias, i.e., original data matrix
	 * is expected.
	 * Note that the columns of W is arranged according to label IDs. Thus
	 * the first column of W will be the projector vector for the label with
	 * ID 0, i.e., the first observed class label. 
	 * 
	 * @param Xt test data matrix with each column being a feature vector
	 * 
	 * @return predicted N x K label matrix, where N is the number of
	 *         test samples, and K is the number of classes
	 *         
	 */
	public RealMatrix predictLabelScoreMatrix(RealMatrix Xt) {
		
		RealMatrix Yt = null;
		if (bias >= 0) {
			// If bias >= 0, W has one row more than Xt,
			// because Xt dosen't have the dummy bias feature 1.
			RealMatrix WPrim = W.getSubMatrix(0, W.getRowDimension() - 2, 0, W.getColumnDimension() - 1);
			Yt = Xt.transpose().multiply(WPrim);
			// biasRowMatrix is 1 x nClass
			RealMatrix biasRowMatrix = W.getRowMatrix(W.getRowDimension() - 1).scalarMultiply(bias);
			int nSample = Xt.getColumnDimension();
			Yt = Yt.add(Matlab.repmat(biasRowMatrix, new int[]{nSample, 1}));
		} else {
			Yt = Xt.transpose().multiply(W);
		}
		return Yt;
		
	}
	
	/**
	 * Predict labels for the test data formated as a original data matrix.
	 * The original data matrix should not has bias features. How do the returned
	 * labels look like depends on model.labels, which is an IDLabelMap.
	 * 
	 * @param Xt test data matrix with each column being a feature vector
	 * 
	 * @return predicted label array with original integer label code
	 * 
	 */
	/*public int[] predict(RealMatrix Xt) {
		RealMatrix Yt = predictLabelScores(Xt);
		int[] pre_labelIDs = labelMatrix2LabelIDArray(Yt);
		int[] IDLabelMap = model.getLabels();
		int[] pre_labels = new int[pre_labelIDs.length];
		for (int i = 0; i < pre_labels.length; i++) {
			pre_labels[i] = IDLabelMap[pre_labelIDs[i]];
		}
		return pre_labels;
	}*/
	
	/**
	 * Predict labels for the test data formated as a feature 2D array.
	 * The original data matrix should not has bias features. How do the returned
	 * labels look like depends on model.labels, which is an IDLabelMap.
	 * 
	 * @param features a 2D feature array
	 * 
	 * @return a label array with original integer label code
	 * 
	 */
	public int[] predict(Feature[][] features) {
		return predict(features2MatrixWithoutBias(features, this.bias));
	}
	
	public void loadSVMModel(String filePath) {
		
		System.out.println("Loading SVM model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			model = (Model)ois.readObject();
			IDLabelMap = model.getLabels();
			ois.close();
			System.out.println("SVM model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		reconstructProjectionMatrix();
		
	}
	
	private void reconstructProjectionMatrix() {
		// IDLabelArray = model.getLabels();
		double[] w = model.getFeatureWeights();
		RealMatrix ww = new Array2DRowRealMatrix(w);
		nClass = model.getNrClass();
		W = Matlab.reshape(ww.transpose(), new int[]{nClass, model.getNrFeature() + 1}).transpose();
	}
	
	public void saveSVMModel(String filePath){

		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(model);
			oos.close();
			System.out.println("SVM model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	@Override
	public void loadModel(String filePath) {
		loadSVMModel(filePath);
	}

	@Override
	public void saveModel(String filePath) {
		saveSVMModel(filePath);
	}
	
}
