import { Asterisk } from "./Asterisk";

type Props = {
  point: string;
  ci?: string;
  asteriskReason?: string;
};

export function MetricCell({ point, ci, asteriskReason }: Props) {
  return (
    <td className="metric">
      <span>
        {point}
        {asteriskReason ? <Asterisk reason={asteriskReason} /> : null}
      </span>
      {ci ? <span className="ci">{ci}</span> : null}
    </td>
  );
}
