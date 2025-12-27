import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function Home() {
  const handleSubmit = () => {
    // Handle the submit action here
  }
  return (
    <>
    <div>
      <Input type="text" placeholder="Enter your question here"/>
      <Button onClick={handleSubmit}>Submit</Button>
    </div>
    </>
  );
}
