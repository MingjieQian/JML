package jml.utils;

import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The <code>Utility</code> class provides some frequently used functions
 * for text processing. 
 * 
 * @author Mingjie Qian
 * @version 1.0, 12/11/2011
 *
 */
public class Utility {
	
	/**
	 * Generic comparator for {@code TreeMap} to sort the keys in a decreasing order.
	 * 
	 * @author Mingjie Qian
	 *
	 * @param <K>
	 *        Class type to be specified by declaration.
	 */
	public static class keyDescendComparator<K extends Comparable<K>> implements Comparator<K> {         
		public int compare(K k1, K k2) {
			return k2.compareTo(k1);  
		}    
	};  
	
	/**
	 * Generic comparator for {@code TreeMap} to sort the keys in a increasing order.
	 * 
	 * @author Mingjie Qian
	 *
	 * @param <K>
	 *        Class type to be specified by declaration.  
	 */
	public static class keyAscendComparator<K extends Comparable<K>> implements Comparator<K> {         
		public int compare(K k1, K k2) {
			return k1.compareTo(k2);  
		}    
	};
	
	/**
	 * Sort a map by its keys according to a specified order. Note: the 
	 * returned map does not allow access by keys. One should use entries
	 * in stead. One can cast the returned map to {@code TreeMap} but not
	 * {@code HashMap}. The input map can be any map.
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend". 
	 * @return
	 *        A sorted {@code TreeMap} by the order specified by {@param order}. 
	 */
	public static <K extends Comparable<K>, V> Map<K, V> sortByKeys(final Map<K, V> map, final String order) {     
		Comparator<K> keyComparator =  new Comparator<K>() {         
			public int compare(K k1, K k2) {
				int compare = 0;
				if ( order.compareTo("descend") == 0 )
					compare = k2.compareTo(k1);
				else if ( order.compareTo("ascend") == 0 )
					compare = k1.compareTo(k2);
				else {
					System.err.println("order should be either \"descend\" or \"ascend\"!");
				}
				if (compare == 0) 
					return 1;       
				else 
					return compare;   
			}
		};     
		Map<K, V> sortedByKeys = new TreeMap<K, V>(keyComparator);
		sortedByKeys.putAll(map);
		return sortedByKeys;
	}
	
	/**
	 * Sort a map by its values according to a specified order. Note: the 
	 * returned map does not allow access by keys. One should use entries
	 * in stead. One can cast the returned map to {@code TreeMap} but not
	 * {@code HashMap}. The input map can be any map.
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend".
	 *        
	 * @return
	 *        A sorted {@code TreeMap} by the order specified by {@param order}. 
	 */
	public static <K, V extends Comparable<V>> Map<K, V> sortByValues(final Map<K, V> map, final String order) {     
		Comparator<K> valueComparator =  new Comparator<K>() {       
			public int compare(K k1, K k2) { 
				int compare = 0;
				if ( order.compareTo("descend") == 0 )
					compare = map.get(k2).compareTo(map.get(k1));
				else if ( order.compareTo("ascend") == 0 )
					compare = map.get(k1).compareTo(map.get(k2));
				else {
					System.err.println("order should be either \"descend\" or \"ascend\"!");
				}
				if (compare == 0) 
					return 1;       
				else 
					return compare;   
			}
		};     
		Map<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
		sortedByValues.putAll(map);
		return sortedByValues;
	}
	
	/**
	 * Sort a map by its values according to a specified order. The input map can be any map. 
	 * One can cast the returned map to {@code HashMap} but not {@code TreeMap}.
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted which can be {@code TreeMap} or {@code HashMap}.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend".
	 * @return
	 *        A sorted {@code TreeMap} by the order specified by {@param order}.
	 */
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(final Map<K, V> map, String order ) {  
		
		List<Map.Entry<K, V>> list =  new LinkedList<Map.Entry<K, V>>( map.entrySet() );   
		
		if ( order.compareTo("ascend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o1.getValue()).compareTo( o2.getValue() );      
				}    
			} );  
		} else if ( order.compareTo("descend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o2.getValue()).compareTo( o1.getValue() );      
				}    
			} ); 
		} else {
			System.err.println("order should be either \"descend\" or \"ascend\"!");
		}
		
		Map<K, V> result = new LinkedHashMap<K, V>();        
		for (Map.Entry<K, V> entry : list) {    
			result.put( entry.getKey(), entry.getValue() );   
		}
		
		return result;    
	}
	
	/**
	 * Sort a map by its keys according to a specified order. The input map can be any map. 
	 * One can cast the returned map to {@code HashMap} but not {@code TreeMap}. 
	 * 
	 * @param <K> 
	 *        Class type for the key in the map.
	 *        
	 * @param <V>
	 *        Class type for the value in the map.
	 *        
	 * @param map
	 *        The map to be sorted which can be {@code TreeMap} or {@code HashMap}.
	 *        
	 * @param order
	 *        The {@code String} indicating the order by which the map
	 *        to be sorted, either "descend" or "ascend".
	 * @return
	 *        A sorted {@code TreeMap} by the order specified by {@param order}. 
	 */
	public static <K extends Comparable<? super K>, V> Map<K, V> sortByKey(final Map<K, V> map, String order ) {  
		
		List<Map.Entry<K, V>> list =  new LinkedList<Map.Entry<K, V>>( map.entrySet() );   
		
		if ( order.compareTo("ascend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o1.getKey()).compareTo( o2.getKey() );      
				}    
			} );  
		} else if ( order.compareTo("descend") == 0 ) {
			Collections.sort( list, new Comparator<Map.Entry<K, V>>() {      
				public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 ) {       
					return (o2.getKey()).compareTo( o1.getKey() );      
				}    
			} ); 
		} else {
			System.err.println("order should be either \"descend\" or \"ascend\"!");
		}
		
		Map<K, V> result = new LinkedHashMap<K, V>();        
		for (Map.Entry<K, V> entry : list) {    
			result.put( entry.getKey(), entry.getValue() );   
		}
		
		return result;    
	}
	
	/**
	 * A generic {@code Class} that implements Comparator<Integer> which provide
	 * a override comparator function sorting a array's indices based on its values.
	 * <p>
	 * Usage:
	 * <code>
	 * <p>
	 * String[] countries = { "France", "Spain", ... };
	 * <p>
	 * ArrayIndexComparator<String> comparator = new ArrayIndexComparator<String>(countries);
	 * <p>
	 * Integer[] idxVector = comparator.createIndexArray();
	 * <p>
	 * Arrays.sort(idxVector, comparator);
	 * </code>
	 * </p>
	 * <p>
	 * Now the indexes are in appropriate order.
	 *
	 * @param <V>
	 *        Class type that extends the {@code Comparable} interface.
	 */
	public static class ArrayIndexComparator<V extends Comparable<? super V>> implements Comparator<Integer> { 
		
		private final V[] array;
		
		public ArrayIndexComparator(V[] array) {
			this.array = array;
		}
		
		public Integer[] createIndexArray() {
			Integer[] idxVector = new Integer[array.length];
			for (int i = 0; i < array.length; i++) {
				idxVector[i] = i; // Autoboxing
			}
			return idxVector;
		}
		
		@Override    
		public int compare(Integer index1, Integer index2) {
			// Autounbox from Integer to int to use as array indexes
			return array[index2].compareTo(array[index1]);
		}
	}

}
