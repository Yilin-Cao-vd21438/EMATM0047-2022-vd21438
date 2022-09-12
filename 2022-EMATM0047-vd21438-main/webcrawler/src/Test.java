import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Test
{
    public static void main( String[] args )
    {
        // regex is even more a pain when multiple lines (unix or windows?)
        // we do not care about lines, so remove lines and blank space
        StringBuilder buf = new StringBuilder();
        In input = new In("example.html");
        while( input.hasNextLine() )
        {
             String line = input.readLine();
             buf.append( line.trim() );
        }
        input.close();

        String collectionsHtml = buf.toString();

        // https://www.javamex.com/tutorials/regular_expressions/multiline.shtml
        //Pattern patternBlurb = Pattern.compile(
        //  "(<li class=\"work blurb group\")(.)?", Pattern.DOTALL | Pattern.UNIX_LINES);
        //Pattern patternBlurb = Pattern.compile("(class=\"work blurb group\")(.)+(</dl>)(</li>)");
        //Pattern patternBlurb = Pattern.compile("<li class=\"work blurb group\"(.)+</dd></dl></li>");

        // PLEASE SEE https://stackoverflow.com/questions/7124778/
        // how-to-match-anything-up-until-this-sequence-of-characters-in-a-regular-expres
        // Using just (.+) will consume everything until last blurb and only return one string
        // Need to tell regex engine to use "non-greedy" version of (.+) by adding the ?
        Pattern patternBlurb = Pattern.compile("(<li class=\"work blurb group\")(.+?)(</dd></dl></li>)");

        //StdOut.println("HTML: "+collectionsHtml);

        int count = 0;

        Matcher matcherBlurb = patternBlurb.matcher(collectionsHtml);
        while( matcherBlurb.find() )
        {
            StdOut.println("BEG ------------------");
            StdOut.println("    BLURB: " + count);
            String blurbHtml = matcherBlurb.group();
            StdOut.println( blurbHtml );
            StdOut.println("    BLURB: " + count);
            StdOut.println("END ------------------");
            count++;
        }
    }

    public static void main1( String[] args )
    {
        //String url = "https://archiveofourown.org/tags/the%20aeon&#39;s%20gate%20series%20-%20sam%20sykes/works";

        String url = "https://archiveofourown.org/tags/absolution%20-%20ramona%20meisel/works";
        StringBuffer eurl = removeUTFCharacters( url );
        System.out.println( eurl.toString() );
    }

    public static StringBuffer removeUTFCharacters(String data)
    {
        System.out.println("ESCAPE: " + data);
        //Pattern p = Pattern.compile("\\\\u(\\p{XDigit}{4})"); // \u2122
        Pattern p = Pattern.compile("\\&#(\\p{XDigit}{2});"); //  &#39;
        Matcher m = p.matcher(data);
        StringBuffer buf = new StringBuffer(data.length());
        while (m.find())
        {
            String ch = String.valueOf((char) Integer.parseInt(m.group(1), 10));
            System.out.println(ch);
            m.appendReplacement(buf, Matcher.quoteReplacement(ch));
        }
        m.appendTail(buf);
        return buf;
    }

}
