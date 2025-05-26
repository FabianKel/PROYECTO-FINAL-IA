import "@/styles/globals.css";
import Head from "next/head";

export default function App({ Component, pageProps }) {
  return (
    <>
      <Head>
        <title>Song Genre Classifier</title>
        <link rel="icon" href="/music.png" />
      </Head>
      <Component {...pageProps} />
    </>
  );
}
