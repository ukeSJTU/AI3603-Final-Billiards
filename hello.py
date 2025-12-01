import pooltool as pt


def main():
    print("This is an example script using pooltool.")
    print("Pooltool version:", pt.__version__)

    # We need a table, some balls, and a cue ball
    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.NINEBALL, table)
    cue = pt.Cue(cue_ball_id="cue")

    # Wrap it up as a System
    shot = pt.System(table=table, balls=balls, cue=cue)

    # Aim at the head ball with a strong impact
    shot.cue.set_state(V0=8, phi=pt.aim.at_ball(shot, "1"))

    # Evolve the shot.
    pt.simulate(shot, inplace=True)

    # Open up the shot in the GUI
    pt.show(shot)


if __name__ == "__main__":
    main()
