Original imagenet label list:
https://raw.githubusercontent.com/tensorflow/models/master/research/tensorrt/labellist.json

Conversion to "one per line" text format:
```
    @Test
    public void temp() throws Exception {
        String out = new Scanner(new URL("https://raw.githubusercontent.com/tensorflow/models/1af55e018eebce03fb61bba9959a04672536107d/research/tensorrt/labellist.json").openStream(), "UTF-8").useDelimiter("\\A").next();

        TypeReference<HashMap<String, String>> typeRef
                = new TypeReference<HashMap<String, String>>() {};
         Map<String,String> m = new ObjectMapper().readValue(out, typeRef);
         int i=0;
         while(m.containsKey(String.valueOf(i))){
             String s = m.get(String.valueOf(i));
             System.out.println(s);
             i++;
         }
    }
```