from bricks_modeling.file_IO.model_reader import read_bricks_from_file

relation_type = {
    '0': "connect",
    '1': "collide",
    '2': "equal",
    '3': "irrelevant"
}

def print_error_msg(relation, eq, collide, connect):
    print(f"Error at relation {relation_type[str(relation)]}!")
    print(f"  connect: ", connect)
    print(f"  collide: ", collide)
    print(f"  equal:  ",eq,"\n")

if __name__ == "__main__":
    folder = "./bricks_modeling/bricks/spacial_relation_test_cases"
    entries = os.listdir(folder)
    entries = [entry for entry in entries if not entry.startswith('.')]
    for entry in entries:
        print("--------------------------")
        brick_path = os.path.join(folder, entry)
        bricks = read_bricks_from_file(brick_path) 
        relation = int(entry[0])

        collide = bricks[0].collide(bricks[1])
        connect = bricks[0].connect(bricks[1]) and (not collide)
        eq = (bricks[0] == bricks[1])

        if relation == 0:
            if not (connect and not eq):
                print_error_msg(relation,eq,collide,connect)
        elif relation == 1:
            if not (collide and not eq):
                print_error_msg(relation,eq,collide,connect)
        elif relation == 2:
            if not (eq and collide):
                print_error_msg(relation,eq,collide,connect)
        elif relation == 3:
            if eq or collide or connect:
                print_error_msg(relation,eq,collide,connect)
            