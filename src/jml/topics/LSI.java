package jml.topics;

import jml.data.Data;
import jml.matlab.Matlab;
import jml.matlab.utils.SingularValueDecompositionImpl;
import jml.options.Options;

import org.apache.commons.math.linear.RealMatrix;


public class LSI extends TopicModel{
	
	public LSI(Options options) {
		super(options.nTopic);
	}
	
	public LSI(int nTopic) {
		super(nTopic);
	}

	@Override
	public void train() {
	
		this.topicMatrix = new SingularValueDecompositionImpl(dataMatrix).getU().
		getSubMatrix(0, dataMatrix.getRowDimension() - 1, 0, nTopic - 1);	
		
	}
	
	public static void main(String[] args) {
		
		String dataMatrixFilePath = "CNN - DocTermCount.txt";
		
		long start = System.currentTimeMillis();
		RealMatrix X = Data.loadMatrixFromDocTermCountFile(dataMatrixFilePath);
		
		X = Matlab.getTFIDF(X);
		X = Matlab.normalizeByColumns(X);
		
		Options options = new Options();
		options.nTopic = 10;
		
		TopicModel topicModel = new LSI(options);
		topicModel.readCorpus(X);
		topicModel.train();
		
		double elapsedTime = (System.currentTimeMillis() - start) / 1000d;
		System.out.println(String.format("Elapsed time: %.3f seconds" + 
				System.getProperty("line.separator") , elapsedTime));
		
	}

}
