package jml.online.classification;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;

public abstract class OnlineBinaryClassifier {
	
	public OnlineBinaryClassifier() {
		W = null;
	}
	
	/**
	 * Projection vector.
	 */
	RealMatrix W;
	
	/**
	 * Train the classifier with a new sample X.
	 */
	public abstract void train(RealMatrix X, int y);
	
	/**
	 * Train the classifier with a new sample X.
	 */
	public void train(double[] X, int y) {
		train(new Array2DRowRealMatrix(X), y);
	}
	
	public abstract int predict(RealMatrix Xt);
	
	public int predict(double[] Xt) {
		return predict(new Array2DRowRealMatrix(Xt));
	}
	
	/**
	 * Load the model for a classifier.
	 * 
	 * @param filePath file path to load the model
	 * 
	 */
	public void loadModel(String filePath) {
		
		System.out.println("Loading model...");
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath));
			W = (RealMatrix)ois.readObject();
			ois.close();
			System.out.println("Model loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Save the model for a classifier.
	 * 
	 * @param filePath file path to save the model
	 * 
	 */
	public void saveModel(String filePath) {
		
		File parentFile = new File(filePath).getParentFile();
		if (parentFile != null && !parentFile.exists()) {
			parentFile.mkdirs();
		}

		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath));
			oos.writeObject(W);
			oos.close();
			System.out.println("Model saved.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

}
