import './globals.css';

export const metadata = {
  title: 'Brainmate — AI Research Assistant',
  description: 'Explore research papers, identify categories, and discover insights with Brainmate.',
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-[#fbfbfb] text-[#18181b] flex flex-col font-sans">
        {children}
      </body>
    </html>
  );
}

