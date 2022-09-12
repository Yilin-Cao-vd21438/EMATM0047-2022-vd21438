import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.io.PrintWriter;

/**
 * Parses HTML, extracts paragraph tags, and writes plain text file.
 *
 * @author James Pope
 */
public class Html2Text {
    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            String u = "  Usage: <HTML file name, not a url | directory name with HTML files>";
            String e = "Example: works-1372.html";
            System.out.println(u);
            System.out.println(e);
            return;
        }


        File f = new File(args[0]);
        if (f.isDirectory()) {
            process(f);
        } else {
            String htmlfile = args[0];

            if (htmlfile.endsWith(".html") == false) {
                String e = "Must specify and HTML file extension";
                throw new IllegalArgumentException(e);
            }

            String textfile = htmlfile.replace(".html", ".txt");
            convert(htmlfile, textfile);
        }

    }


    /**
     * Parses the HTML file paragraph taqs and writes to text file.
     *
     * @param htmlfile
     * @param textfile
     * @throws IOException
     */
    public static void convert(String htmlfile, String textfile)
            throws IOException {
        System.out.printf("Processing %s...\n", htmlfile);

        Document doc = Jsoup.parse(new File(htmlfile), "UTF8");
        //Document doc = Jsoup.connect(url).get();
        Elements paragraphs = doc.select("p");

        FileWriter fw = new FileWriter(textfile);
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter out = new PrintWriter(bw);

        //System.out.printf("\nParagraph: (%d)\n", paragraphs.size());
        for (Element src : paragraphs) {
            // Found most (hopefully all) author text to be in tag with
            // class "userstuff".  It may not always be a div and seems 
            // to be nested, so look up through the parents.
            // This also eliminates the "Notes" at the end which is good.
            // Unfortunately, we still get header like follows:
            // 
            boolean userstuff = false;
            Elements parents = src.parents();
            for (Element parent : parents) {
                /*
                 * blockquote will also has class="userstuff" but these are
                 * comments by the author not the book.  I believe (but have
                 * not completely verified) that the book will always have
                 * div tag with class="userstuff"
                 */
                if (parent.tagName().equalsIgnoreCase("div") && parent.hasClass("userstuff")) {
                    //Parent userstuff parent blockquote for p
                    //Parent userstuff parent blockquote for p
                    //Parent userstuff parent blockquote for p
                    //Parent userstuff parent div for p
                    //Parent userstuff parent div for p
                    //...
                    //System.out.printf("Parent %s for %s\n", parent.tagName(), src.tagName());
                    userstuff = true;
                    break;
                }
            }
            //if( src.hasAttr("userstuff") )
            //if( src.hasClass("userstuff") )
            if (userstuff && src.hasText()) {
                //System.out.printf(" * %s: <%s>", src.tagName(), src.text());
                //System.out.printf("%s\n", src.text());
                out.println(src.text());
            }
        }

        out.close();
        bw.close();
        fw.close();
    }

    public static void process(File dir) throws IOException {
        File[] files = dir.listFiles();
        for (File file : files) {
            String filename = file.getPath();
            if (file.isFile() && filename.endsWith(".html")) {
                String textfile = filename.replace(".html", ".txt");
                convert(filename, textfile);
            } else if (file.isDirectory()) {
                process(file);
            }
        }
    }
}
