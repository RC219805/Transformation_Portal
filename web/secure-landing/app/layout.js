export const dynamic = "force-dynamic";

export const metadata = {
  title: "Dynamic Neural Access",
  description: "Managed front door for the Transformation Portal operator console."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
