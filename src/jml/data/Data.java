package jml.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import jml.matlab.Matlab;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.OpenMapRealMatrix;
import org.apache.commons.math.linear.RealMatrix;


import static jml.matlab.Matlab.display;
import static jml.matlab.Matlab.fprintf;
import static jml.matlab.Matlab.full;
import static jml.matlab.Matlab.norm;
import static jml.matlab.Matlab.sparse;

/**
 * All indices of docTermCountArray start from 1.
 */
public class Data {

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		String dataMatrixFilePath = "";
		String dirMatlab = "";
		// dataMatrixFilePath = "CNN - DocTermCount.txt";
		dataMatrixFilePath = "sampled_data";
		
		long start = System.currentTimeMillis();
		RealMatrix X = loadMatrixFromDocTermCountFile(dataMatrixFilePath);
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.println(String.format("Elapsed time: %.3f seconds" + 
				System.getProperty("line.separator") , elapsedTime));
		
		X = X.getSubMatrix(0, 6, 0, 6);
		Matlab.printMatrix(X);
		X = Matlab.getTFIDF(X);
		Matlab.printMatrix(X);
		X = Matlab.normalizeByColumns(X);
		Matlab.printMatrix(X);
				
		String denseMatrixFileName = "denseMatrix.txt";
		saveDenseMatrix(denseMatrixFileName, X);
		
		RealMatrix D = loadDenseMatrix(denseMatrixFileName);
		Matlab.printMatrix(D);
		
		fprintf("|| X - D ||_F: %f\n", norm(X.subtract(D)));
		
		// Read a dense matrix from a text file generate by MATLAB
		dirMatlab = "C:\\Aaron\\My Codes\\Matlab\\IO for Java";
		String denseMatrixFilePath = dirMatlab + File.separator + denseMatrixFileName;
		D = loadDenseMatrix(denseMatrixFilePath);
		Matlab.printMatrix(D);
		
		String sparseMatrixFileName = "sparseMatrix.txt";
		saveSparseMatrix(sparseMatrixFileName, X);
		
		RealMatrix S = loadSparseMatrix(sparseMatrixFileName);
		Matlab.printMatrix(S);
		
		// Read a sparse matrix from a text file generate by Matlab
		dirMatlab = "C:\\Aaron\\My Codes\\Matlab\\IO for Java";
		String sparseMatrixFilePath = dirMatlab + File.separator + sparseMatrixFileName;
		D = loadSparseMatrix(sparseMatrixFilePath);
		Matlab.printMatrix(D);
		
		// Test the uniform interface for data communication
		RealMatrix M = loadDenseMatrix(denseMatrixFilePath);
		display(M);
		
		// Write a dense matrix into a text file
		String fileName = "denseMatrix2.txt";
		saveMatrix(fileName, M);

		// Read a dense matrix from a text file
		M = loadMatrix(fileName);
		display(M);

		// Write a sparse matrix into a text file
		fileName = "sparseMatrix2.txt";
		saveMatrix(fileName, sparse(M));

		// Read a sparse matrix from a text file
		M = loadMatrix(fileName);
		display(M);
		display(full(M));
		
	}
	
	/**
	 * Write a matrix into a text file. Sparseness will be automatically detected.
	 * 
	 * @param dataMatrixFilePath file path to write a matrix into
	 * 
	 * @param A a real matrix
	 * 
	 */
	public static void saveMatrix(String dataMatrixFilePath, RealMatrix A) {
		
		if (A.getClass().getSimpleName().equals("OpenMapRealMatrix")) {
			saveSparseMatrix(dataMatrixFilePath, A);
		} else {
			saveDenseMatrix(dataMatrixFilePath, A);
		}
		
	}
	
	/**
	 * Read a matrix from a text file. Sparseness will be automatically detected.
	 * 
	 * @param dataMatrixFilePath file path to read a matrix from
	 * 
	 * @return a real matrix
	 * 
	 */
	public static RealMatrix loadMatrix(String dataMatrixFilePath) {
		
		RealMatrix M = null;
		Pattern pattern = null;
		// pattern = Pattern.compile("[(]([\\d]+), ([\\d]+)[)][:]? ([-\\d.]+)");
		pattern = Pattern.compile("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)");
		Matcher matcher = null;
		
		BufferedReader br = null;
		try {
            br = new BufferedReader(new FileReader(dataMatrixFilePath));
        } catch (FileNotFoundException e) {
			System.out.println("Cannot open file: " + dataMatrixFilePath);
			e.printStackTrace();
		}
		
		String line = "";
		try {
			while ((line = br.readLine()) != null) {
				if (line.startsWith("#"))
					continue;
				matcher = pattern.matcher(line);
				break;
			}
			br.close();
			if (matcher.find()) {
				M = loadSparseMatrix(dataMatrixFilePath);
			} else {
				M = loadDenseMatrix(dataMatrixFilePath);
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return M;

	}
	
	/**
	 * Write a sparse matrix into a text file. Each line 
     * corresponds to a non-zero entry with the format "(%d, %d) %.8g".
     * 
	 * @param dataMatrixFilePath file path to write a sparse matrix into
	 * 
	 * @param A a sparse matrix
	 * 
	 */
	public static void saveSparseMatrix(String dataMatrixFilePath, RealMatrix A) {
		
		PrintWriter pw = null;
		
		try {
			pw = new PrintWriter(new FileWriter(dataMatrixFilePath));
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + dataMatrixFilePath);
		}
		
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		
		int sampleID = 0;
		int featureID = 0;

		for (int j = 0; j < nCol; j++) {
			sampleID = j + 1;
			for (int i = 0; i < nRow; i++) {
				featureID = i + 1;
				if (A.getEntry(i, j) != 0) {
					pw.printf("(%d, %d) %.8g" + System.getProperty("line.separator") , featureID, sampleID, A.getEntry(i, j));
		        }
			}
		}
		
		if (!pw.checkError()) {
			pw.close();
			System.out.println("Data matrix file written: " + dataMatrixFilePath + System.getProperty("line.separator"));
		} else {
			pw.close();
			System.err.println("Print stream has encountered an error!");
		}

	}
	
	/**
	 * Load a {@code RealMatrix} from a doc-term-count file located at
	 * {@code String} docTermCountFilePath.
	 * 
	 * @param docTermCountFilePath
     *        a {@code String} specifying the location of the doc-term-count 
     *        file holding matrix data. Each line is an entry with the format
     *        (sampleID,[whitespace]featureID):[whitespace]value". sampleID
     *        and featureID start from 1.
     *        
     * @return a sparse matrix
	 */
	public static RealMatrix loadMatrixFromDocTermCountFile(String docTermCountFilePath) {
		
		Pattern pattern = null;
		String line;
		BufferedReader br = null;
		Matcher matcher = null;
		TreeMap<Integer, Double> feature = null;
		int sampleID_old = 0;
		int sampleID = 0;
		int featureID = 0;
		double value = 0;
		int nSample = 0;
		int nFeature = 0;		
		// int nz = 0;
				
		pattern = Pattern.compile("[(]?([\\d]+)[,]? ([\\d]+)[)]?[:]? ([-\\d.]+)");
		
        try {

            br = new BufferedReader(new FileReader(docTermCountFilePath));
            
        } catch (FileNotFoundException e) {
			
			System.out.println("Cannot open file: " + docTermCountFilePath);
			e.printStackTrace();
			
		} 
        
        ArrayList<TreeMap<Integer, Double>> featureArray = new ArrayList<TreeMap<Integer, Double>>();
        
		try {
			while ((line = br.readLine()) != null) {
				
				if (line.startsWith("#"))
					continue;

				matcher = pattern.matcher(line);

				if (!matcher.find()) {
					System.out.println("Data format for the docTermCountFile should be: (sampleID, featureID): value");
					System.exit(0);
				}

				sampleID = Integer.parseInt(matcher.group(1));
				if (sampleID != sampleID_old) {
					if (nSample > 0) {
						
						featureArray.add(feature);
						if (nFeature < feature.lastKey().intValue()) {
							nFeature = feature.lastKey().intValue();
						}
						// System.out.println("sampleID: " + sampleID_old + ", nUniqueTerms: " + feature.size());
						
						/*
						 * Sometime a document may have empty content after filtered
						 * by stop words. 
						 */
						for (int i = sampleID_old + 1; i < sampleID; i++) {
							featureArray.add(new TreeMap<Integer, Double>());
							System.out.println("sampleID: " + ++nSample + ", Empty");
						}
						
					}
					for (int i = nSample + 1; i < sampleID; i++) {
						featureArray.add(new TreeMap<Integer, Double>());
						System.out.println("sampleID: " + ++nSample + ", Empty");					
					}
					nSample++;
					feature = new TreeMap<Integer, Double>();
				}
				featureID = Integer.parseInt(matcher.group(2));
				value = Double.parseDouble((matcher.group(3)));

				feature.put(featureID, value);
				sampleID_old = sampleID;
				// nz++;
			}

			if (feature != null) {
				featureArray.add(feature);
				// System.out.println("sampleID: " + sampleID_old + ", nUniqueTerms: " + feature.size());
			}
			br.close();
		} catch (NumberFormatException e) {

			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		}
		
		// System.out.println();
		
		return mapArray2Matrix(featureArray);
		
	}
	
	/**
     * Load a {@code RealMatrix} from a text file located at {@code String} dataMatrixFilePath.
     * 
     * @param dataMatrixFilePath
     *        a {@code String} specifying the location of the text file holding matrix data.
     *        Each line is an entry with the format (without double quotes) 
     *        "(rowIdx,[whitespace]colIdx):[whitespace]value". rowIdx and colIdx
     *        start from 1 as in MATLAB.
     *        
     * @return a sparse matrix
     * 
     */
	public static RealMatrix loadSparseMatrix(String dataMatrixFilePath) {
		
		Pattern pattern = null;
		String line;
		BufferedReader br = null;
		Matcher matcher = null;
		TreeMap<Integer, Double> feature = null;
		int sampleID_old = 0;
		int sampleID = 0;
		int featureID = 0;
		double value = 0;
		int nSample = 0;
		int nFeature = 0;		
		// int nz = 0;
				
		pattern = Pattern.compile("[(]([\\d]+)[,] ([\\d]+)[)][:]? ([-\\d.]+)");
		
        try {

            br = new BufferedReader(new FileReader(dataMatrixFilePath));
            
        } catch (FileNotFoundException e) {
			
			System.out.println("Cannot open file: " + dataMatrixFilePath);
			e.printStackTrace();
			
		}
        
        ArrayList<TreeMap<Integer, Double>> featureArray = new ArrayList<TreeMap<Integer, Double>>();
        
		try {
			while ((line = br.readLine()) != null) {
				
				if (line.startsWith("#"))
					continue;

				matcher = pattern.matcher(line);

				if (!matcher.find()) {
					System.out.println("Data format for the docTermCountFile should be: (sampleID, featureID): value");
					System.exit(0);
				}

				sampleID = Integer.parseInt(matcher.group(2));
				if (sampleID != sampleID_old) {
					if (nSample > 0) {
						
						featureArray.add(feature);
						if (nFeature < feature.lastKey().intValue()) {
							nFeature = feature.lastKey().intValue();
						}
						// System.out.println("sampleID: " + sampleID_old + ", nUniqueTerms: " + feature.size());
						
						/*
						 * Sometime a document may have empty content after filtered
						 * by stop words. 
						 */
						for (int i = sampleID_old + 1; i < sampleID; i++) {
							featureArray.add(new TreeMap<Integer, Double>());
							System.out.println("sampleID: " + ++nSample + ", Empty");
						}
						
					}
					for (int i = nSample + 1; i < sampleID; i++) {
						featureArray.add(new TreeMap<Integer, Double>());
						System.out.println("sampleID: " + ++nSample + ", Empty");					
					}
					nSample++;
					feature = new TreeMap<Integer, Double>();
				}
				featureID = Integer.parseInt(matcher.group(1));
				value = Double.parseDouble((matcher.group(3)));

				feature.put(featureID, value);
				sampleID_old = sampleID;
				// nz++;
			}

			if (feature != null) {
				featureArray.add(feature);
				// System.out.println("sampleID: " + sampleID_old + ", nUniqueTerms: " + feature.size());
			}
			br.close();
		} catch (NumberFormatException e) {

			e.printStackTrace();
		} catch (IOException e) {

			e.printStackTrace();
		}
		
		// System.out.println();
		
		return mapArray2Matrix(featureArray);
	
	}

	public static RealMatrix mapArray2Matrix(
			ArrayList<TreeMap<Integer, Double>> featureArray) {
		
		int nFeature = 0;
		int lastKey = 0;
		int sampleID = 0;
		// int cnt = 0;
		int nSample = 0;
		
		// Get feature size
		Iterator<TreeMap<Integer, Double>> iter = featureArray.iterator();
		TreeMap<Integer, Double> feature = null;
		for (TreeMap<Integer, Double> fea : featureArray) {
			// cnt++;
			// out.println(cnt);
			if (!fea.isEmpty() && nFeature < (lastKey = fea.lastKey())) {
				nFeature = lastKey;
			}
		}
		nFeature = nFeature > 0 ? nFeature : 1;
		
		// Get corpus size
		nSample = featureArray.size();
		
		if (nSample == 0) {
			System.out.println("Empty data matrix, nothing loaded!");
			return null;
		}
		
		// Assign dataMatrix from featureArray
		RealMatrix dataMatrix = new OpenMapRealMatrix(nFeature, nSample);
		iter = featureArray.iterator();
		nSample = 0;
		while (iter.hasNext()) {
			
			feature = iter.next();
			/*if (feature.isEmpty()) {
				continue;
			}*/
			sampleID++;
			for (int featureID : feature.keySet()) {
				dataMatrix.setEntry(featureID - 1, sampleID - 1, feature.get(featureID));
			}
			
		}
		
		return dataMatrix;
	}
	
	/**
	 * Write a dense matrix into a text file. Each line 
     * corresponds to a row with the format "%.8g\t%.8g\t%.8g\t... \t%.8g".
     * 
	 * @param dataMatrixFilePath file path to write a dense matrix into
	 * 
	 * @param A a dense matrix
	 * 
	 */
	public static void saveDenseMatrix(String dataMatrixFilePath, RealMatrix A) {
		
		PrintWriter pw = null;
		
		try {
			pw = new PrintWriter(new FileWriter(dataMatrixFilePath));
		} catch (IOException e) {
			System.out.println("IO error for creating file: " + dataMatrixFilePath);
		}
		
		int nRow = A.getRowDimension();
		int nCol = A.getColumnDimension();
		
		StringBuilder strBuilder = new StringBuilder(200);
		for (int i = 0; i < nRow; i++) {
			strBuilder.setLength(0);
			for (int j = 0; j < nCol; j++) {
				strBuilder.append(String.format("%.8g\t", A.getEntry(i, j)));
			}
			// pw.printf("%g\t", A.getEntry(i, j));
			pw.println(strBuilder.toString().trim());
		}
		
		if (!pw.checkError()) {
			pw.close();
			System.out.println("Data matrix file written: " + dataMatrixFilePath + System.getProperty("line.separator"));
		} else {
			pw.close();
			System.err.println("Print stream has encountered an error!");
		}
		
	}
	
	/**
	 * Read a dense matrix from a text file. Each line 
     * corresponds to a row with the format "%.8g\t%.8g\t%.8g\t... \t%.8g".
     * 
	 * @param dataMatrixFilePath file path to read a dense matrix from
	 * 
	 * @return a dense matrix
	 * 
	 */
	public static RealMatrix loadDenseMatrix(String dataMatrixFilePath) {
		
		BufferedReader textIn = null;
		
		try {
			textIn = new BufferedReader(// Read text from a character-input stream
					new InputStreamReader(// Read bytes and decodes them into characters 
							new FileInputStream(dataMatrixFilePath)));// Read bytes from a file
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		String line = null;

		ArrayList<double[]> denseArr = new ArrayList<double[]>();
		try {
			while ((line = textIn.readLine()) != null) {
				
				String[] strArr = line.split("[\t ]");
				double[] vec = new double[strArr.length];
				for (int i = 0; i < strArr.length; i++) {
					vec[i] = Double.parseDouble(strArr[i]);
				}
				denseArr.add(vec);
			}
			
			try {
				textIn.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return TwoDimArray2Matrix(denseArr);
		
	}
	
	
	public static RealMatrix TwoDimArray2Matrix(ArrayList<double[]> denseArr) {
		int nRow = denseArr.size();
		double[][] data = new double[nRow][];
		for (int i = 0; i < nRow; i++) {
			data[i] = denseArr.get(i);
		}
		return new Array2DRowRealMatrix(data);
	}
	
	
	public static RealMatrix docTermCountArray2Matrix(
			ArrayList<TreeMap<Integer, Integer>> docTermCountArray) {

		int nDoc = 0;
		int nTerm = 0;
		int lastKey = 0;
		int docID = 0;
		int cnt = 0;
		
		// Get the vocabulary size
		Iterator<TreeMap<Integer, Integer>> iter = docTermCountArray.iterator();
		TreeMap<Integer, Integer> feature = null;
		
		while (iter.hasNext()) {
			feature = iter.next();
			cnt++;
			System.out.println(cnt);
			if (!feature.isEmpty() && nTerm < (lastKey = feature.lastKey())) {
				nTerm = lastKey;
			}
		}
		nTerm = nTerm > 0 ? nTerm : 1;
		
		// Get corpus size
		nDoc = docTermCountArray.size();

		if (nDoc == 0) {
			System.out.println("Empty data matrix, nothing loaded!");
			return null;
		}
		
		// Assign docTermCountMatrix from docTermCountArray
		RealMatrix docTermCountMatrix = new OpenMapRealMatrix(nTerm, nDoc);
		iter = docTermCountArray.iterator();
		while (iter.hasNext()) {
			feature = iter.next();
			docID++;
			for (int termID : feature.keySet()) {
				docTermCountMatrix.setEntry(termID - 1, docID - 1, (double)feature.get(termID));
			}
		}
		
		return docTermCountMatrix;
		
	}

}
