public class Prgm9 {
	static class Example <T>{
		T ob;
		void setitem(T o) {
			ob=o;
		}
		public T getitem() {
			return ob;
		}
	}
	
	public static <K> void display(K[] arr) {
		for(K a:arr) {
			System.out.println(a+" ");
		}
		
	}
	public static void main(String[] args) {

		Example<Integer> T=new Example<Integer>();
		
		T.setitem(5);
		System.out.println("Generic getitem: "+T.getitem());
	
		Integer[] intA= {5,6,7};
		Float[] floA= {(float) 5.5, (float) 6.6, (float) 7.7};
		Character[] chaA= {'a','b','b'};
		
		System.out.println("Integer Array:");
		display(intA);
		System.out.println("Float Array:");
		display(floA);
		System.out.println("Character Array:");
		display(chaA);
	}
}
